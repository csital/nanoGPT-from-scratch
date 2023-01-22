import subprocess
from pathlib import Path

from path_definitions import REPOSITORY_ROOT

DOWNLOAD_FOLDER = REPOSITORY_ROOT / "downloads"


def download_file_from_url(url: str, overwrite: bool = False) -> Path:
    """Use wget command to download file from url.

    Args:
        url (str): url string.
        overwrite (bool, optional): Overwrite the file if it already exists. Defaults to False.

    Returns:
        Path: filepath to downloaded file.
    """
    filename = url.rsplit("/", 1)[1]
    output_filepath = DOWNLOAD_FOLDER / filename

    # do not download file if it already exists
    # unless overwrite is specified
    if not (output_filepath).is_file() or overwrite:
        # wait until process is finished to continue
        subprocess.Popen(f"wget -nv -P {str(DOWNLOAD_FOLDER)} {url}", shell=True).wait()

    assert output_filepath.is_file(), f'Failed to download file from "{url}"!'

    return output_filepath
