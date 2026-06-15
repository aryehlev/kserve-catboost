/// Fetch the official KServe gRPC proto at build time, falling back to the
/// committed copy when network access is unavailable (offline builds, CI, etc.).
///
/// Pinned source — update the tag when adopting a new KServe release:
///   https://github.com/kserve/kserve/blob/v0.14.1/docs/predict-api/v2/grpc_predict_v2.proto
const OFFICIAL_PROTO_URL: &str =
    "https://raw.githubusercontent.com/kserve/kserve/v0.14.1/docs/predict-api/v2/grpc_predict_v2.proto";

const BUNDLED_PROTO_PATH: &str = "proto/grpc_predict_v2.proto";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Re-run whenever the committed fallback or the offline flag changes.
    println!("cargo:rerun-if-changed={BUNDLED_PROTO_PATH}");
    println!("cargo:rerun-if-env-changed=KSERVE_PROTO_OFFLINE");

    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
    let fetched = std::path::PathBuf::from(&out_dir).join("grpc_predict_v2.proto");

    // Choose the proto source: fetched (pinned upstream) or bundled (fallback).
    // We write to OUT_DIR rather than overwriting the tracked proto file so
    // builds are reproducible and the working tree stays clean.
    let (proto, include): (std::path::PathBuf, std::path::PathBuf) =
        if std::env::var("KSERVE_PROTO_OFFLINE").is_ok() {
            println!("cargo:warning=KSERVE_PROTO_OFFLINE set — using bundled proto");
            (BUNDLED_PROTO_PATH.into(), "proto".into())
        } else {
            match fetch_proto_to(&fetched) {
                Ok(()) => (fetched, out_dir.into()),
                Err(e) => {
                    println!("cargo:warning=Failed to fetch proto ({e}). Using bundled copy.");
                    (BUNDLED_PROTO_PATH.into(), "proto".into())
                }
            }
        };

    tonic_build::configure()
        .build_server(true)
        .build_client(false)
        .compile_protos(&[proto], &[include])?;

    Ok(())
}

fn fetch_proto_to(dest: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    let body = ureq::get(OFFICIAL_PROTO_URL).call()?.into_string()?;
    std::fs::write(dest, body)?;
    Ok(())
}

/// Try to download the latest official KServe proto and overwrite the bundled
/// copy.  Failures are demoted to cargo warnings so the build still succeeds
/// using the committed fallback.
fn fetch_official_proto() {
    match ureq::get(OFFICIAL_PROTO_URL).call() {
        Ok(resp) => match resp.into_string() {
            Ok(body) => {
                if let Err(e) = std::fs::write(PROTO_PATH, body) {
                    println!(
                        "cargo:warning=Could not write fetched proto to {PROTO_PATH}: {e}"
                    );
                } else {
                    println!(
                        "cargo:warning=Updated {PROTO_PATH} from official KServe source"
                    );
                }
            }
            Err(e) => {
                println!(
                    "cargo:warning=Failed to read proto response body: {e}. Using bundled copy."
                );
            }
        },
        Err(e) => {
            println!(
                "cargo:warning=Failed to fetch official KServe proto ({e}). Using bundled copy."
            );
        }
    }
}
