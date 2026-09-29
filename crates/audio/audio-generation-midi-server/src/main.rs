use clap::Parser;

#[derive(Debug, Parser)]
#[command(
    name = "audio-generation-midi-server",
    version,
    about = "Thin HTTP API adapter for audio-generation-midi"
)]
struct Args {
    /// Address to bind, for example 127.0.0.1:3000.
    #[arg(long, default_value = "127.0.0.1:3000")]
    addr: String,
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();
    eprintln!(
        "audio-generation-midi-server listening on http://{}",
        args.addr
    );
    audio_generation_midi_server::serve(&args.addr)
}
