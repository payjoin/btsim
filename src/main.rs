use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    num_wallets: usize,
    #[arg(short, long)]
    payment_obligations: usize,
    #[arg(short, long)]
    time_steps: usize,
    #[arg(short, long)]
    block_interval: usize,
    #[arg(short, long)]
    seed: Option<u64>,
    #[arg(short, long)]
    payment_obligation_deadline_threshold: f64,
}

fn main() {
    let args = Args::parse();
    let mut sim = btsim::SimulationBuilder::new(
        args.seed.unwrap_or(42),
        args.num_wallets,
        args.time_steps,
        args.block_interval,
        args.payment_obligations,
        args.payment_obligation_deadline_threshold,
    )
    .build();

    sim.build_universe();
    let result = sim.run();
    result.save_tx_graph("graph.svg");
    println!(
        "Total payment obligations: {}",
        result.total_payment_obligations()
    );
    println!(
        "Missed payment obligations percentage: {:?}",
        result.precentage_of_payment_obligations_missed()
    );
}
