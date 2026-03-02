use std::env;

fn main() {
    env_logger::init();

    // Read config file path from environment or use default
    let config_path = env::var("CONFIG_FILE").unwrap_or_else(|_| "config.toml".to_string());

    let config = btsim::config::Config::from_file(&config_path)
        .expect(&format!("Failed to parse config file: {}", config_path));

    let seed = config.simulation.seed.unwrap_or(42);
    let mut sim = btsim::SimulationBuilder::new(
        seed,
        config.wallet_types,
        config.simulation.max_timestep,
        1, // TODO: hardcoded block interval for now. If we change this we need to ensure payment obligations are not being double handled.
        config.simulation.num_payment_obligations,
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
        result.percentage_of_payment_obligations_missed()
    );
}
