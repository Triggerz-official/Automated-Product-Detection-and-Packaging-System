import pandas as pd
import requests
import cv2
import numpy as np
import easyocr
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import io
import time

# Global EasyOCR reader to avoid reinitializing for each image
reader = easyocr.Reader(['en'])

def download_image(url):
    response = requests.get(url)
    return response.content

def extract_text(image_bytes):
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    result = reader.readtext(img)
    return ' '.join([text for _, text, _ in result])

def process_row(row):
    image_url = row['image_link']
    try:
        image_bytes = download_image(image_url)
        text = extract_text(image_bytes)
        return text
    except Exception as e:
        return f"Error processing {image_url}: {str(e)}"

def process_batch(batch):
    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_row = {executor.submit(process_row, row): row for row in batch.to_dict('records')}
        for future in as_completed(future_to_row):
            results.append(future.result())
    return results

def main():
    # Read the CSV file
    total_entries = 100
    chunk_size = 20  # Process in smaller chunks
    output_file = "train_with_extracted_text_100.csv"
    
    # Write the header to the output file
    pd.DataFrame(columns=['image_link', 'extracted_text']).to_csv(output_file, index=False)
    
    start_time = time.time()
    
    with tqdm(total=total_entries, desc="Processing images") as pbar:
        for chunk in pd.read_csv("sample_test.csv", chunksize=chunk_size):
            if len(chunk) > total_entries - pbar.n:
                chunk = chunk.iloc[:total_entries - pbar.n]
            
            extracted_texts = process_batch(chunk)
            chunk['extracted_text'] = extracted_texts
            
            # Append the results to the CSV file
            chunk.to_csv(output_file, mode='a', header=False, index=False)
            
            pbar.update(len(chunk))
            
            if pbar.n >= total_entries:
                break

    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"Processing complete. Results saved to '{output_file}'")
    print(f"Total processing time: {processing_time:.2f} seconds")
    print(f"Average time per entry: {processing_time/total_entries:.2f} seconds")

if __name__ == "__main__":
    main()