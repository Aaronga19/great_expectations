"""AI Notebooks Jupyter Service configuration file."""

import logging
import os
import requests
from requests.adapters import HTTPAdapter
from jupyter_client import kernelspec

# pylint: disable=anomalous-backslash-in-string, line-too-long, undefined-variable
c.NotebookApp.open_browser = False
c.ServerApp.token = ""
c.ServerApp.password = ""
c.ServerApp.port = 8080
c.ServerApp.allow_origin_pat = "(^https://8080-dot-[0-9]+-dot-devshell\.appspot\.com$)|(^https://colab\.research\.google\.com$)|((https?://)?[0-9a-z]+-dot-datalab-vm[\-0-9a-z]*.googleusercontent.com)|((https?://)?[0-9a-z]+-dot-[\-0-9a-z]*.notebooks.googleusercontent.com)|((https?://)?[0-9a-z\-]+\.[0-9a-z\-]+\.cloudshell\.dev)|((https?://)ssh\.cloud\.google\.com/devshell)"
c.ServerApp.allow_remote_access = True
c.ServerApp.disable_check_xsrf = False
c.ServerApp.notebook_dir = "/home/jupyter"
# pylint: enable=anomalous-backslash-in-string, line-too-long, undefined-variable

BASE_PATH = "/opt/deeplearning/metadata/"
MAX_RETRIES = 2
METADATA_URL = "http://metadata/computeMetadata/v1"
METADATA_FLAVOR = {"Metadata-Flavor": "Google"}


def _get_session(prefix="http://", max_retries=MAX_RETRIES):
  """Return an HTTP Session.

  Args:
    prefix(str): Prefix for URL
    max_retries(int): Maximum number of retries each connection should attempt.

  Returns:
    A requests.Session()
  """
  session = requests.Session()
  session.mount(prefix, HTTPAdapter(max_retries=max_retries))
  return session


def get_metadata_value(metadata_key):
  """Get Metadata value.

  Args:
    metadata_key(str): Metadata key to look in Compute Metadata.

  Returns:
    A Metadata value or None
  """
  if metadata_key is None:
    raise ValueError("Invalid metadata key")
  try:
    session = _get_session(max_retries=5)
    response = session.get(
      f"{METADATA_URL}/instance/attributes/{metadata_key}",
      headers=METADATA_FLAVOR,
    )
    response.raise_for_status()
    print(f"Metadata {metadata_key}:{response.text}")
    return response.text
  except requests.exceptions.HTTPError as err:
    if err.response.status_code == 404:
      print(err)
  return None


def read_from_file(path):
  """Read metadata file.

  Args:
    path(str) Location of file with metadata information.

  Returns:
    A string.
  """
  with open(path, "r", encoding="utf-8") as file:
    return file.read().replace("\n", "")


def get_env_name():
  return read_from_file(os.path.join(BASE_PATH, "env_version"))


def get_env_uri():
  return read_from_file(os.path.join(BASE_PATH, "env_uri"))


local_kernelspec_cache = {}
def metadata_env_pre_save(model, **kwargs):  # pylint: disable=unused-argument
  """Save metadata from Jupyter Environment.

  Args:
    model(dict): Notebooks information
  """

  try:
    # only run on notebooks
    if model["type"] != "notebook":
      return
    # only run on nbformat v4 or later
    if model["content"]["nbformat"] < 4:
      return
    model_metadata = model["content"]["metadata"]
    if "kernelspec" in model_metadata:
      kernel = model_metadata["kernelspec"]["name"]
      # remote kernels have no compatible container at the moment
      if kernel.startswith("remote-"):
        del model_metadata["kernelspec"]
        model_metadata["environment"] = {
            "type": "gcloud",
            "name": get_env_name(),
        }
        return
      # local kernels will have the local prefix in managed notebooks
      if kernel.startswith("local-"):
        kernel = kernel.split("-", 1)[1]
        if kernel not in local_kernelspec_cache:
          for k in kernelspec.find_kernel_specs():
            local_kernelspec_cache[k] = kernelspec.get_kernel_spec(k)
        kernel_metadata = local_kernelspec_cache[kernel].metadata
        model_metadata["environment"] = {
            "type": "gcloud",
            "name": get_env_name(),
            "uri": kernel_metadata["google.kernel_container"],
            # local name may not match kernel name on container
            "kernel": kernel_metadata["google.kernel_name"],
        }
        return
      # non-managed notebooks should have correct kernelspec listed
      model_metadata["environment"] = {
          "type": "gcloud",
          "name": get_env_name(),
          "uri": get_env_uri(),
          "kernel": kernel,
      }
  # pylint: disable=broad-except
  except (FileNotFoundError, KeyError, OSError, Exception) as e:
    logging.error("Failed to enrich the Notebook with metadata: %s", e)

# pylint: disable=undefined-variable
c.FileContentsManager.pre_save_hook = metadata_env_pre_save

# https://jupyterlab.readthedocs.io/en/stable/user/rtc.html
if get_metadata_value('use-collaborative'):
  print('Using JupyterLab Collaborative flag')
  c.LabApp.collaborative = True
c.ServerApp.notebook_dir = '/home/jupyter'
