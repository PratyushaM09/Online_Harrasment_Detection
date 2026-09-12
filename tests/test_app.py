from app import READY_MESSAGE, main


def test_main_prints_ready_message(capsys):
    main()

    captured = capsys.readouterr()
    assert captured.out.strip() == READY_MESSAGE

