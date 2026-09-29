use clap::{Parser, Subcommand};
use runtime_core::cli::read_json_input;

#[derive(Debug, Parser)]
#[command(
    name = "audio-analysis-processing-cli",
    version,
    about = "Thin CLI adapter for audio-analysis-processing"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Print package and adapter metadata.
    Info {
        /// Emit machine-readable JSON.
        #[arg(long)]
        json: bool,
    },
    /// Print the command schema.
    Schema {
        /// Emit machine-readable JSON.
        #[arg(long)]
        json: bool,
    },
    /// Print library operations.
    Operations {
        /// Emit machine-readable JSON.
        #[arg(long)]
        json: bool,
    },
    /// Run one library-owned operation.
    Run {
        /// Operation id.
        #[arg(long, default_value = "describe")]
        operation: String,
        /// JSON request payload.
        #[arg(long)]
        json: Option<String>,
        /// Read JSON request payload from a file.
        #[arg(long)]
        file: Option<String>,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    match cli.command.unwrap_or(Command::Info { json: false }) {
        Command::Info { json } => print_payload(
            json,
            "audio-analysis-processing",
            &audio_analysis_processing_cli::package_metadata_json(),
        ),
        Command::Schema { json } => print_payload(
            json,
            "audio-analysis-processing command schema",
            &audio_analysis_processing_cli::command_schema_json(),
        ),
        Command::Operations { json } => {
            let payload = serde_json::to_string(
                &audio_analysis_processing_cli::package_surface().operations,
            )?;
            print_payload(json, "audio-analysis-processing operations", &payload);
        }
        Command::Run {
            operation,
            json,
            file,
        } => {
            let input = read_json_input(json, file)?;
            let response = audio_analysis_processing_cli::run_operation(&operation, input)
                .map_err(std::io::Error::other)?;
            println!("{}", serde_json::to_string(&response)?);
        }
    }
    Ok(())
}

fn print_payload(json: bool, title: &str, payload: &str) {
    if json {
        println!("{payload}");
    } else {
        println!("{title}");
        println!("{payload}");
    }
}
