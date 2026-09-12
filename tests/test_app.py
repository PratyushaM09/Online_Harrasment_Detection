from app import READY_MESSAGE, main
from src.config import load_config


def test_main_prints_ready_message_and_config_summary(capsys):
    main()

    captured = capsys.readouterr()
    config = load_config()
    expected_output = "\n".join(
        [
            READY_MESSAGE,
            f"Project: {config.project.name}",
            f"Labels configured: {len(config.labels)}",
        ]
    )

    assert captured.out.strip() == expected_output
