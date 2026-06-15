/// Fetch the official KServe gRPC proto at build time, falling back to the
/// committed copy when network access is unavailable (offline builds, CI, etc.).
///
/// Source:
///   https://raw.githubusercontent.com/kserve/kserve/master/docs/predict-api/v2/grpc_predict_v2.proto
const OFFICIAL_PROTO_URL: &str =
    "https://raw.githubusercontent.com/kserve/kserve/master/docs/predict-api/v2/grpc_predict_v2.proto";

const PROTO_PATH: &str = "proto/grpc_predict_v2.proto";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Re-run this script whenever the proto file changes.
    println!("cargo:rerun-if-changed={PROTO_PATH}");
    println!("cargo:rerun-if-env-changed=KSERVE_PROTO_OFFLINE");

    // Skip the network fetch when explicitly requested (e.g. hermetic builds).
    if std::env::var("KSERVE_PROTO_OFFLINE").is_ok() {
        println!("cargo:warning=KSERVE_PROTO_OFFLINE set — using bundled proto");
    } else {
        fetch_official_proto();
    }

    tonic_build::configure()
        .build_server(true)
        .build_client(false)
        .compile_protos(&[PROTO_PATH], &["proto"])?;

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
