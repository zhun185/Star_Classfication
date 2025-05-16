import csv
import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin, urlencode
import io # Required for pre-processing CSV lines

# --- Configuration ---
CSV_FILE = 'Skyserver_SQL5_9_2025 11_46_25 AM.csv'  # Your input CSV file
BASE_SPECTRA_DIR = 'spectra'  # Base directory for spectra
BASE_IMAGES_DIR = 'images'    # Base directory for images
MAX_WORKERS = 10  # Number of parallel downloads for spectra/images combined
TIMEOUT_SECONDS = 30  # Timeout for each download request

# SDSS SAS Base URLs for Spectra (DR17 and fallbacks)
SAS_BASE_URLS = [
    "https://data.sdss.org/sas/dr17/eboss/spectro/redux/v5_13_0/spectra/lite/",
    "https://data.sdss.org/sas/dr17/boss/spectro/redux/v5_13_0/spectra/lite/",
    "https://data.sdss.org/sas/dr17/sdss/spectro/redux/26/spectra/lite/",
    "https://data.sdss.org/sas/dr16/eboss/spectro/redux/v5_13_0/spectra/lite/",
    "https://data.sdss.org/sas/dr12/boss/spectro/redux/v5_7_0/spectra/lite/",
    "https://data.sdss.org/sas/dr12/sdss/spectro/redux/26/spectra/lite/",
]

# SDSS Image Cutout Service URL
IMAGE_CUTOUT_BASE_URL = "https://skyserver.sdss.org/dr17/SkyServerWS/ImgCutout/getjpeg"
IMAGE_WIDTH_PX = 256
IMAGE_HEIGHT_PX = 256
IMAGE_SCALE_ARCSEC_PER_PIXEL = 0.396 # Default SDSS scale

# --- Helper Functions ---

def construct_sas_urls(plate_val, mjd_val, fiberid_val): # Renamed args for clarity
    urls = []
    try:
        # Ensure inputs are integers for formatting
        plate = int(plate_val)
        mjd = int(mjd_val)
        fiberid = int(fiberid_val)

        mjd_str = str(mjd)
        fiberid_for_file = f"{fiberid:04d}" # Standard 4-digit padded fiber for filename
    except ValueError:
        print(f"Warning: Invalid plate, mjd, or fiberid format for URL construction: p={plate_val}, m={mjd_val}, f={fiberid_val}")
        return []

    for base_url in SAS_BASE_URLS:
        plate_for_path_segment: str
        plate_for_filename_segment: str

        if "/sdss/" in base_url: # Legacy SDSS (e.g., run2d=26 for DR17 path)
            plate_for_path_segment = f"{plate:04d}"  # Plate dir is 4-digit padded
            plate_for_filename_segment = f"{plate:04d}" # Plate in filename is 4-digit padded
        elif "/eboss/" in base_url or "/boss/" in base_url:
            plate_for_path_segment = str(plate) # Plate dir is NOT padded
            plate_for_filename_segment = str(plate) # Plate in filename is NOT padded
        else:
            # Fallback for any other SAS_BASE_URLS that don't match /sdss/, /eboss/, /boss/
            # Defaulting to non-padded as it's common for newer surveys.
            # Add a warning if this is unexpected.
            print(f"Warning: Unknown SAS URL type for plate padding: {base_url}. Defaulting to non-padded plate.")
            plate_for_path_segment = str(plate)
            plate_for_filename_segment = str(plate)

        # Construct the filename: spec-[plate]-[mjd]-[fiber].fits
        filename = f"spec-{plate_for_filename_segment}-{mjd_str}-{fiberid_for_file}.fits"
        
        # Construct the full URL path
        # The base URLs in SAS_BASE_URLS already include ".../spectra/lite/"
        url = urljoin(base_url, f"{plate_for_path_segment}/{filename}")
        if url not in urls: # Avoid adding duplicate URLs if logic somehow produces them
            urls.append(url)
            
    return urls # No need for list(set(urls)) if duplicates are avoided during append

def construct_image_url(ra, dec):
    try:
        ra_float = float(ra)
        dec_float = float(dec)
    except (ValueError, TypeError):
        return None # Invalid input
    params = {
        'ra': ra_float,
        'dec': dec_float,
        'width': IMAGE_WIDTH_PX,
        'height': IMAGE_HEIGHT_PX,
        'scale': IMAGE_SCALE_ARCSEC_PER_PIXEL
    }
    return f"{IMAGE_CUTOUT_BASE_URL}?{urlencode(params)}"

def download_file(url, output_filepath, file_type_label):
    """Downloads a single file (spectrum or image)."""
    if not url:
        return False, f"{file_type_label} URL was None, skipping download."
    try:
        print(f"Attempting to download {file_type_label}: {url}")
        response = requests.get(url, timeout=TIMEOUT_SECONDS, stream=True)
        if response.status_code == 200:
            with open(output_filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Successfully downloaded: {output_filepath} from {url}")
            return True, f"{file_type_label} Downloaded: {output_filepath}"
        elif response.status_code == 404:
            print(f"{file_type_label} not found (404) at {url}.")
            return False, f"{file_type_label} Not Found (404): {url}"
        else:
            print(f"Failed to download {file_type_label} from {url}: Status {response.status_code}")
            return False, f"Failed {file_type_label} (Status {response.status_code}): {url}"
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {file_type_label} from {url}: {e}")
        return False, f"Error downloading {file_type_label}: {e}"

def process_object_data(row_data, row_number):
    """
    Downloads spectrum and image for a single object.
    Returns status messages and an identifier.
    """
    obj_id_for_log = f"Row {row_number} (specobjid: {row_data.get('specobjid', 'N/A')})"

    try:
        # Validate and extract necessary data first
        plate_raw = row_data.get('plate')
        mjd_raw = row_data.get('mjd')
        fiberid_raw = row_data.get('fiberid')
        ra_raw = row_data.get('ra')
        dec_raw = row_data.get('dec')
        subclass_raw = row_data.get('subclass', "").strip()
        specobjid = row_data.get('specobjid', "").strip() # Keep as string for filename

        if not all([plate_raw, mjd_raw, fiberid_raw, specobjid]): # specobjid is crucial for filename
            return f"Skipping {obj_id_for_log} due to missing plate, mjd, fiberid, or specobjid.", "Image download not attempted due to missing essential spec/obj data.", obj_id_for_log, True
        
        # Attempt conversions after ensuring keys exist and have values
        # These will be passed to URL construction functions which handle their own int/float conversion
        
        subclass = subclass_raw.split(' ')[0].replace('/', '_') if subclass_raw else "Unknown"
        # Use original specobjid for filename, ensure it's not empty if it passed the check above
        if not specobjid: specobjid = f"unknown_obj_row_{row_number}" 

    except KeyError as e: # Should be caught by .get() but as a safeguard
        return f"Skipping {obj_id_for_log} due to initial missing key: {e}", "Image download not attempted due to missing key.", obj_id_for_log, True
    # ValueError for int/float conversions will be handled within construct_sas_urls and construct_image_url

    # --- Spectrum Download --- 
    spectra_output_subdir = os.path.join(BASE_SPECTRA_DIR, subclass)
    os.makedirs(spectra_output_subdir, exist_ok=True)
    spectrum_filename = f"spec-{specobjid}.fits"
    spectrum_output_filepath = os.path.join(spectra_output_subdir, spectrum_filename)
    
    spectrum_status = f"Spectrum Failed for {obj_id_for_log}"
    spectrum_downloaded_this_run = False

    if os.path.exists(spectrum_output_filepath):
        spectrum_status = f"Spectrum Exists: {spectrum_output_filepath}"
    else:
        # Pass raw string values, conversion to int happens in construct_sas_urls
        possible_spectrum_urls = construct_sas_urls(plate_raw, mjd_raw, fiberid_raw)
        if not possible_spectrum_urls:
            spectrum_status = f"Spectrum Failed for {obj_id_for_log} (no valid URLs generated, check plate/mjd/fiberid values or format in CSV: p={plate_raw}, m={mjd_raw}, f={fiberid_raw})."
        else:
            for url_idx, url in enumerate(possible_spectrum_urls):
                print(f"Trying spectrum URL {url_idx+1}/{len(possible_spectrum_urls)} for {obj_id_for_log}: {url}")
                success, status_msg = download_file(url, spectrum_output_filepath, "Spectrum")
                spectrum_status = status_msg # Keep last status
                if success:
                    spectrum_downloaded_this_run = True
                    break 
            if not spectrum_downloaded_this_run and "Not Found" in spectrum_status:
                 spectrum_status = f"Spectrum Failed for {obj_id_for_log} (404 from all URLs)."

    # --- Image Download --- 
    images_output_subdir = os.path.join(BASE_IMAGES_DIR, subclass)
    os.makedirs(images_output_subdir, exist_ok=True)
    image_filename = f"img-{specobjid}.jpg"
    image_output_filepath = os.path.join(images_output_subdir, image_filename)

    image_status = f"Image Failed for {obj_id_for_log}"

    if not ra_raw or not dec_raw: # Check if ra/dec were present for URL construction
        image_status = f"Image download skipped for {obj_id_for_log} (missing ra or dec data in CSV)."
    elif os.path.exists(image_output_filepath):
        image_status = f"Image Exists: {image_output_filepath}"
    else:
        # Pass raw string values, conversion to float happens in construct_image_url
        image_url = construct_image_url(ra_raw, dec_raw)
        if image_url:
            success, status_msg = download_file(image_url, image_output_filepath, "Image")
            image_status = status_msg
        else:
            image_status = f"Image download skipped for {obj_id_for_log} (invalid ra/dec for URL)."
            
    return spectrum_status, image_status, obj_id_for_log, "Skipping" in spectrum_status or "Skipping" in image_status

def main():
    if not os.path.exists(CSV_FILE):
        print(f"Error: CSV file '{CSV_FILE}' not found.")
        return

    os.makedirs(BASE_SPECTRA_DIR, exist_ok=True)
    os.makedirs(BASE_IMAGES_DIR, exist_ok=True)
    
    rows_to_process = []
    header = None
    cleaned_csv_lines = []

    try:
        with open(CSV_FILE, mode='r', encoding='utf-8') as infile:
            for line in infile:
                stripped_line = line.strip()
                if not stripped_line:  # Skip empty lines
                    continue
                if stripped_line.startswith('#'): # Skip comment lines
                    print(f"Skipping comment line: {stripped_line}")
                    continue
                cleaned_csv_lines.append(stripped_line)
        
        if not cleaned_csv_lines:
            print("Error: CSV file contains no data after stripping comments and empty lines.")
            return

        # Use the first non-comment, non-empty line as header for DictReader
        # Then feed the rest of the lines
        # Wrap lines in a StringIO object to be read by DictReader
        csvfile_string_io = io.StringIO('\n'.join(cleaned_csv_lines))
        reader = csv.DictReader(csvfile_string_io)
        
        # Check if fieldnames were successfully parsed
        if not reader.fieldnames:
            print("Error: Could not determine CSV header. Please check CSV format.")
            return
        print(f"CSV Header identified as: {reader.fieldnames}")

        for i, row in enumerate(reader):
            if not any(val and val.strip() for val in row.values()): # if all values in the row are empty or effectively empty
                print(f"Skipping effectively empty row {i+1} (after header): {row}")
                continue
            rows_to_process.append(row)
            
    except Exception as e:
        print(f"Error reading or processing CSV file: {e}")
        return

    if not rows_to_process:
        print("No data rows found in CSV to process after cleaning.")
        return

    stats = {
        'spectra_downloaded': 0, 'spectra_failed': 0, 'spectra_existed': 0,
        'images_downloaded': 0, 'images_failed': 0, 'images_existed': 0,
        'skipped_due_to_bad_data': 0, 'total_objects_in_csv': len(rows_to_process)
    }

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Pass row_number (index + 1, assuming header is not counted in rows_to_process enumeration)
        future_to_log_id = {executor.submit(process_object_data, row, i+1): f"Row {i+1}" for i, row in enumerate(rows_to_process)}
        
        for i, future in enumerate(as_completed(future_to_log_id)):
            log_id = future_to_log_id[future]
            try:
                s_status, i_status, obj_id_from_func, was_skipped = future.result()
                print(f"Processed {obj_id_from_func} ({i+1}/{stats['total_objects_in_csv']}):\n  Spectrum: {s_status}\n  Image:    {i_status}")
                
                if was_skipped:
                    stats['skipped_due_to_bad_data'] += 1
                else:
                    if "Downloaded:" in s_status: stats['spectra_downloaded'] += 1
                    elif "Exists:" in s_status: stats['spectra_existed'] += 1
                    elif "Failed" in s_status : stats['spectra_failed'] += 1 # Catch various fail reasons

                    if "Downloaded:" in i_status: stats['images_downloaded'] += 1
                    elif "Exists:" in i_status: stats['images_existed'] += 1
                    elif "Failed" in i_status or "Error" in i_status or "Not Found" in i_status or "skipped" in i_status : stats['images_failed'] +=1

            except Exception as exc:
                print(f"Critical error processing task for {log_id}: {exc}")
                stats['skipped_due_to_bad_data'] += 1 # Count as skipped if future itself errors

    print("\n--- Download Summary ---")
    print(f"Total objects in CSV (after cleaning): {stats['total_objects_in_csv']}")
    print(f"Skipped due to invalid/missing essential data: {stats['skipped_due_to_bad_data']}")
    print("-- Spectra --")
    print(f"  Successfully downloaded: {stats['spectra_downloaded']}")
    print(f"  Already existed: {stats['spectra_existed']}")
    print(f"  Failed to download (after attempting): {stats['spectra_failed']}")
    print("-- Images --")
    print(f"  Successfully downloaded: {stats['images_downloaded']}")
    print(f"  Already existed: {stats['images_existed']}")
    print(f"  Failed or skipped (after attempting or due to data issues): {stats['images_failed']}")
    print(f"\nSpectra saved in: '{BASE_SPECTRA_DIR}'")
    print(f"Images saved in:    '{BASE_IMAGES_DIR}'")

if __name__ == "__main__":
    main()