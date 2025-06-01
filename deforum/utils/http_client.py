import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# Returns a retrying http client. This is low overhead
# so fine to retrieve on every request.
def get_http_client(
    retries=5, 
    backoff_factor=0.3,
    status_forcelist=[429, 500, 502, 503, 504],
    session=None,
) -> requests.Session :
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session