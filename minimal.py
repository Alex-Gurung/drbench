# suppress warnings
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Disable logging for specific libraries
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

from drbench import drbench_enterprise_space, task_loader
from drbench.agents.drbench_agent.drbench_agent import DrBenchAgent
from drbench.config import RunConfig, set_run_config
from drbench.score_report import score_report

if __name__ == "__main__":
    # Configure models
    agent_model = "openrouter/openai/gpt-4o-mini"
    embedding_model = "openrouter/openai/text-embedding-ada-002"
    evaluation_model = "openrouter/openai/gpt-4o"

    # Ensure a canonical run_dir for default-on logging.
    repo_root = Path(__file__).resolve().parent
    run_dir = repo_root / "runs" / f"minimal_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    set_run_config(RunConfig(model="gpt-4o-mini", run_dir=run_dir))

    # (1) Load one task
    # ----------------------
    task = task_loader.get_task_from_id(task_id="DR0001")
    print(task.summary())

    # (2) Start DRBench Enterprise Search Environment
    # ----------------------
    env = drbench_enterprise_space.DrBenchEnterpriseSearchSpace(
        task=task.get_path(),
        start_container=True,
        # auto_ports = True,  # Uncomment to use random ports
    )

    # (3) Generate Report with Your Own Agent
    # ----------------------
    dr_agent = DrBenchAgent(
        model=agent_model,
        max_iterations=5,
        embedding_model=embedding_model,
    )

    report = dr_agent.generate_report(
        query=task.get_task_config()["dr_question"],
        env=env,
    )

    # (4) Evaluate Report
    # ----------------------
    score_dict = score_report(
        predicted_report=report,
        task=task,
        metrics=["insights_recall", "factuality"],
        savedir="results/minimal",
        model=evaluation_model,
    )
    print("Insights Recall: ", score_dict["insights_recall"])
    print("Factuality: ", score_dict["factuality"])

    # (5) Exit
    # ----------------------
    env.delete()
