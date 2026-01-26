# suppress warnings
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

from drbench import task_loader
from drbench.agents.drbench_agent.drbench_agent import DrBenchAgent
from drbench.agents.basic_agent import BasicAgent
from drbench.config import RunConfig, set_run_config
from drbench.score_report import score_report


if __name__ == "__main__":
    """
    This Evaluate the Ability for an Agent to Generate a DR Report directly from the Local Files.
    """
    # Provide a run_dir for default-on logging.
    repo_root = Path(__file__).resolve().parent
    run_dir = repo_root / "runs" / f"minimal_local_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    set_run_config(RunConfig(run_dir=run_dir))

    # (1) Load one task
    # ----------------------
    task = task_loader.get_task_from_id(task_id="SANITY0")
    print(task.summary())

    dr_question = task.get_task_config()["dr_question"]
    task_local_files = task.get_local_files_list()

    # (2) Generate Report with Your Own Agent
    # ----------------------
    dr_agent = BasicAgent()

    # Report is a dict with 'report_text' and 'report_insights'
    report = dr_agent.generate_report(query=dr_question, local_files=task_local_files)

    # (3) Evaluate Report
    # ----------------------
    score_dict = score_report(
        predicted_report=report,
        task=task,
        metrics=["insights_recall"],
        savedir="results/minimal_local",
    )
    print("Insights Recall: ", score_dict["insights_recall"])
