use clap::Parser;

#[derive(Debug, Parser)]
#[command(
    name = "audio-analysis-io-server",
    version,
    about = "Thin HTTP API adapter for audio-analysis-io"
)]
struct Args {
    /// Address to bind, for example 127.0.0.1:3000.
    #[arg(long, default_value = "127.0.0.1:3000")]
    addr: String,
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();
    eprintln!("audio-analysis-io-server listening on http://{}", args.addr);
    audio_analysis_io_server::serve(&args.addr)
}
