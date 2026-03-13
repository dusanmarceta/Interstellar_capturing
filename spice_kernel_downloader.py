import os
import urllib.request
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_kernel(url, filename):
    if os.path.exists(filename):
        print(f"[*] Found {filename} locally. Skipping.")
        return
        
    print(f"[*] Downloading {filename}...")
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
        urllib.request.urlretrieve(url, filename, reporthook=t.update_to)
    print(f"[+] Finished downloading {filename}\n")


print("=== SPICE Kernel Download ===\n")

# 1. Leapseconds (Time conversion)
download_kernel(
    "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls", 
    "naif0012.tls"
)

# 2. Planetary Constants (Masses)
download_kernel(
    "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/gm_de440.tpc", 
    "gm_de440.tpc"
)

# 3. Ephemeris Part 2 (Positions/Velocities from 1969 AD to 17191 AD)
# NOTE: This file is ~1.6 GB. It will take a moment depending on your connection.
download_kernel(
    "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de441_part-2.bsp", 
    "de441_part-2.bsp"
)

print("=== All kernels downloaded! ===")