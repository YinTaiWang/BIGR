import os
import cv2
import re
import argparse

####################
###  Parse_args  ###
####################
def parse_args():
    """
        Parses inputs from the commandline.
        :return: inputs as a Namespace object
    """
    parser = argparse.ArgumentParser(description="Settings to create segmentation learning progress video.")
    parser.add_argument("-f", "--folder", help="The folder name of the result.",
                        required=True)
    parser.add_argument("-i", "--file_ID", help="The file ID. (e.g. epoch1_22.png, file ID = 22.",
                        default=22, required=False)
    parser.add_argument("-n", "--video_name", help="File name for the video. Default = video", 
                        default='video',required=False)

    return parser.parse_args()

def sort_key(name):
        numbers = re.findall(r'\d+', name)
        return [int(num) for num in numbers]

def main():
    ####################
    ## Process arguments
    args = parse_args()
    folder = args.folder
    i = str(args.file_ID)
    video_name = str(args.video_name)

    script_dir = os.getcwd()
    result_folder = os.path.join(script_dir, "out", folder)
    image_folder = os.path.join(result_folder, "val_check")

    images = [img for img in os.listdir(image_folder) if img.endswith(f"_{i}.png")]
    print(f"number of images: {len(images)}")

    
    # Sort the filenames using the custom key
    images = sorted(images, key=sort_key)

    # Video Properties
    video_name = os.path.join(result_folder, f"{video_name}.mp4")
    frame = cv2.imread(os.path.join(image_folder, images[0]))  # Read the first image to get the width and height
    height, width, layers = frame.shape

    # Video Writer
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'X264'), 15, (width, height))

    # Add images to video
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()
    print(f"{video_name} created!")
    
if __name__ == "__main__":
    main()
