from learning_to_presolve import CONFIGS, generate_packing, oracle_configuration

problem = generate_packing(n_vars=28, n_constraints=9, seed=23)
oracle = oracle_configuration(problem)

print(f"oracle configuration: {oracle.selected}")
for config, evaluation in zip(CONFIGS, oracle.evaluations, strict=True):
    print(
        config,
        {
            "objective": round(evaluation.objective, 4),
            "nodes": evaluation.nodes,
            "residual_size": evaluation.residual_vars + evaluation.residual_rows,
            "checks": evaluation.checks,
            "work_proxy": round(evaluation.work_proxy, 3),
        },
    )
