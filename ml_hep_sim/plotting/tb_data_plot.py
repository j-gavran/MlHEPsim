from pathlib import Path

from tensorflow.python.summary.summary_iterator import summary_iterator


def get_tb_data(dir_path, to_extract):
    path = Path(dir_path).glob("**/*")

    event_files = [x for x in path if x.is_file()]
    for f in event_files:
        if ".out" in str(f):
            event_file = str(f)
            break

    values = [[] for _ in range(len(to_extract))]

    for event in summary_iterator(event_file):
        for value in event.summary.value:
            for i, extract in enumerate(to_extract):
                if value.tag == extract:
                    values[i].append(value.simple_value)

    return values
