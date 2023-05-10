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

    parser.add_argument('-t',
                        '--target',
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
    
    parser.add_argument('-f',
                        '--file-extension',
                        help='File extension for images (e.g. jpg)',
                        metavar='file-extension',
                        type=str,
                        default='jpg')

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

def get_images(path: pathlib.Path, img_ext: str = 'jpg') -> t.List[str]:
    """Get list of images

    Args:
        path (pathlib.Path): Path to data
        img_ext (str, optional): Image file extension. Defaults to 'jpg'.

    Returns:
        t.List[str]: List of image files
    """
    images = []
    # Get list of files
    for root, dir, files in os.walk(path):
        # exclude hidden dirs and files
        files = [file for file in files if not file.startswith('.') and not os.path.basename(file).startswith('.') and file.endswith(img_ext)]
        images += [os.path.join(root, file) for file in files]
    
    return list(set(images))

def extract_zip(data: pathlib.Path, img_ext: str = 'jpg') -> t.List[str]:
    """Extract zip file

    If the files are already extracted, this will just return the list of files.

    Args:
        data (pathlib.Path): Path to zip file
        img_ext (str, optional): Image file extension. Defaults to 'jpg'.

    Returns:
        List[str]: List of extracted files
    """
    # Extract zip file
    with zipfile.ZipFile(data, 'r') as zip_ref:
        zipped_files = zip_ref.filelist
        # now we can test if the files are already extracted
        if all([os.path.exists(data.parent / 'extracted' / z.filename) for z in zipped_files]):
            # if they are, just return the list of files
            print('Files already extracted previously')
        else:
            # otherwise, extract them
            print('Extracting files')
            zip_ref.extractall(data.parent / 'extracted')
        
    # Get list of files
    files = get_images(data.parent, img_ext)
    return files

def load_data(data_path: pathlib.Path, image_ext: str = 'jpg') -> t.Tuple[t.List[np.ndarray], t.List[str]]:
    """Load image data

    Args:
        data_path (pathlib.Path): Path to image data

    Returns:
        t.Tuple(t.List[np.ndarray] t.List[str]): List of images and list of filenames

    """
    # If data_path is a directory, get all the files in that directory
    if data_path.is_dir():
        files = get_images(data_path, image_ext)
    elif data_path.is_file() and data_path.suffix == '.zip':
        files = extract_zip(data_path, image_ext)
    # If there's still only one (or zero) files, it's probably not a valid path
    if len(list(files)) <= 1:
        raise ValueError(f'"{data_path}" is not a valid path')
    
    images = []
    for image_file in files:
        image = parse_image(image_file)
        images.append(image)

    return images, files

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
                        names: t.List[str],
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
    sorted_distances = sorted(zip(names, distances), key=lambda x: x[1])
    # Return n+1 most similar images (the first one will be the target image)
    return sorted_distances[:min(n+1, len(sorted_distances))]

def main():
    """Main function"""
    # Get command-line arguments
    args = get_args()

    # Load image data
    images, image_names = load_data(args.data, args.file_extension)
    
    # if no target is specified, prompt with a range (min, max indices of images)
    if args.target is None:
        target = None
        while not type(target) == np.ndarray:
            min_index = 1
            max_index = len(images)
            print('No target image specified. Either specify a target image with the -t flag or enter a number below to select a target image from the data directory.')
            print('Please specify a target image:')
            print(f'Enter a number between {min_index} and {max_index}')
            target_index = input()
            if not target_index.isdigit():
                raise ValueError('Please enter a number')
            target_index = int(target_index)
            if target_index < min_index or target_index > max_index:
                raise ValueError(f'Please enter a number between {min_index} and {max_index}')
            target = images[target_index-1]
            target_name = image_names[target_index-1]
    elif type(args.target) == str:
        args.target = args.data / args.target
        if not args.target.is_file():
            raise ValueError(f'"{args.target}" is not a valid path')
        target = parse_image(args.target)
    else:
        raise ValueError(f'"{args.target}" is not a valid path')
    
    # Find similar images
    similar_images = find_similar_images(target, images, image_names, args.num_similar)

    # Save results to CSV file
    df = pd.DataFrame(similar_images, columns=['filename', 'distance'])
    df.to_csv(args.out / 'similar_images.csv', index=False)

if __name__ == '__main__':
    main()