"""simple-image-search.py: Code for cds-visual assignment 1

For this assignment, you'll be using ```OpenCV``` to design a simple image search algorithm.

The dataset is a collection of over 1000 images of flowers, sampled from 17 different species.

For this exercise, you should write some code which does the following:

- Define a particular image that you want to work with
- For that image
  - Extract the colour histogram using ```OpenCV```
- Extract colour histograms for all of the **other* images in the data
- Compare the histogram of our chosen image to all of the other histograms 
  - For this, use the ```cv2.compareHist()``` function with the ```cv2.HISTCMP_CHISQR``` metric
- Find the five images which are most simlar to the target image
  - Save a CSV file to the folder called ```out```, showing the five most similar images and the distance metric:

|Filename|Distance]
|---|---|
|target|0.0|
|filename1|---|
|filename2|---|
"""

import typing as t
import os
import zipfile
import pathlib
import argparse
import cv2
import numpy as np
import pandas as pd

def get_args():
    """Get command-line arguments

    Returns:
        argparse.Namespace: command-line arguments
    """
    parser = argparse.ArgumentParser(
        description='Find similar images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('target',
                        metavar='target',
                        type=str,
                        help='Target image (filename in data)')

    parser.add_argument('-d',
                        '--data',
                        help='Path to image data (can be a zip file)',
                        metavar='data',
                        type=pathlib.Path,
                        default='data')

    parser.add_argument('-o',
                        '--out',
                        help='Path to output directory',
                        metavar='out',
                        type=pathlib.Path,
                        default='out')
    
    parser.add_argument('-n',
                        '--num-similar',
                        help='Number of similar images to return',
                        metavar='num-similar',
                        type=int,
                        default=5)

    return parser.parse_args()

def parse_image(filename: str) -> np.ndarray:
    """Parse image

    Args:
        filename (str): Path to image

    Returns:
        np.ndarray: Image
    """
    # Read image
    image = cv2.imread(filename)
    # Convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def extract_zip(data: pathlib.Path) -> t.List[str]:
    """Extract zip file

    Args:
        data (pathlib.Path): Path to zip file

    Returns:
        List[str]: List of extracted files
    """
    # Extract zip file
    with zipfile.ZipFile(data, 'r') as zip_ref:
        zip_ref.extractall(data.parent / 'extracted')
    # Get list of extracted files
    files = data.parent.glob('*')
    return files

def load_data(data_path: pathlib.Path) -> t.List[np.ndarray]:
    """Load image data

    Args:
        data_path (pathlib.Path): Path to image data

    Returns:
        t.List[np.array]: List of images
    """
    # If data_path is a directory, get all the files in that directory
    if data_path.is_dir():
        files = data_path.glob('*')
    # If data_path is a file, assume it's a zip file and extract it
    elif data_path.is_file() and data_path.suffix == '.zip':
        files = extract_zip(data_path)
    else:
        raise ValueError(f'"{data_path}" is not a valid path')

    # Load images
    images = []
    for image_file in files:
        image = parse_image(image_file)
        images.append(image)

    return images

def get_histogram(image: np.ndarray) -> np.ndarray:
    """Get histogram of image

    Args:
        image (np.ndarray): Image

    Returns:
        np.ndarray: Histogram
    """
    # Calculate histogram
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],
                        [0, 256, 0, 256, 0, 256])
    # Normalize histogram
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def compare_histograms(target_hist: np.ndarray,
                       hist: np.ndarray) -> float:
    """Compare two histograms

    Args:
        target_hist (np.ndarray): Target histogram
        hist (np.ndarray): Histogram

    Returns:
        float: Distance between histograms
    """
    # Compare histograms
    distance = cv2.compareHist(target_hist, hist, cv2.HISTCMP_CHISQR)
    return distance

def find_similar_images(target: np.ndarray,
                        images: t.List[np.ndarray],
                        n: int = 5) -> t.List[t.Tuple[str, float]]:
    """Find n most similar images

    Args:
        target (np.ndarray): Target image
        images (t.List[np.ndarray]): List of images
        n (int, optional): Number of similar images to return. Defaults to 5.

    Returns:
        t.List[t.Tuple[str, float]]: List of (filename, distance) tuples
    """
    # Get histogram of target image
    target_hist = get_histogram(target)
    # Get histogram of all other images
    hists = [get_histogram(image) for image in images]
    # Compare target histogram to all other histograms
    distances = [compare_histograms(target_hist, hist) for hist in hists]
    # Sort by distance
    sorted_distances = sorted(zip(images, distances), key=lambda x: x[1])
    # Return n most similar images
    return sorted_distances[:n]

def main():
    """Main function
    """
    # Get command-line arguments
    args = get_args()

    # Load target image
    target = parse_image(args.target)

    # Load image data
    images = load_data(args.data)

    # Find similar images
    similar_images = find_similar_images(target, images, args.num_similar)

    # Save results to CSV file
    df = pd.DataFrame(similar_images, columns=['filename', 'distance'])
    df.to_csv(args.out / 'similar_images.csv', index=False)

if __name__ == '__main__':
    main()