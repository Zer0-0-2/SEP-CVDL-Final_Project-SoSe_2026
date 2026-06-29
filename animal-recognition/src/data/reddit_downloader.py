import os
import subprocess

# Destination path to store images for training CNNs
# Using the existing animal-recognition/data directory
DESTINATION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))

# List containing the subreddits
SUBREDDITS = {
    "Abyssinian": "https://www.reddit.com/r/Abyssinians/",
    "Bengal": "https://www.reddit.com/r/bengalcats/",
    "Birman": "https://www.reddit.com/r/birmans/",
    "Bombay": "https://www.reddit.com/r/BombayCat/",
    "British_Shorthair": "https://www.reddit.com/r/britishshorthair/",
    "Maine_Coon": "https://www.reddit.com/r/mainecoons/",
    "Ragdoll": "https://www.reddit.com/r/ragdollcats/",
    "Sphynx": "https://www.reddit.com/r/SphynxCats/",
    "Tabby": "https://www.reddit.com/r/TabbyCats/",
    # "Tiger_Cat": "TODO: find some other source", see tiger_cat_downloader.py,
    "Beagle": "https://www.reddit.com/r/beagles/",
    "Pug": "https://www.reddit.com/r/pugs/",
    "Boxer": "https://www.reddit.com/r/Boxer/",
    "Shiba_Inu": "https://www.reddit.com/r/shiba/",
    "Samoyed": "https://www.reddit.com/r/samoyeds/",
    "Golden_Retriever": "https://www.reddit.com/r/goldenretrievers/",
    "German_Shepherd": "https://www.reddit.com/r/germanshepherds/",
    "Siberian_Husky": "https://www.reddit.com/r/siberianhusky/",
    "Dalmatian": "https://www.reddit.com/r/dalmatians/",
    "Rottweiler": "https://www.reddit.com/r/Rottweiler/"
}

import subprocess

def download_reddit_images():
    os.makedirs(DESTINATION_DIR, exist_ok=True)
    print(f"Saving images to: {DESTINATION_DIR}")

    for class_name, base_url in SUBREDDITS.items():
        print(f"\n--- Downloading images for {class_name} ---")
        target_urls = [
            f"{base_url}top/?t=all",
            f"{base_url}top/?t=year",
            f"{base_url}top/?t=month",
        ]
        class_dest_path = os.path.join(DESTINATION_DIR, class_name)
        os.makedirs(class_dest_path, exist_ok=True)

        # Skip immediately if we already have 200+ images
        current_images = [f for f in os.listdir(class_dest_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        if len(current_images) >= 200:
            print(f"Already have {len(current_images)} images for {class_name}. Skipping to next.")
            continue

        try:
            for img_num in range(1, 10):
                # Check how many images we already have before each fallback pass
                current_images = [f for f in os.listdir(class_dest_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
                if len(current_images) >= 200:
                    break

                print(f"Fetching image index {img_num} from galleries for {class_name}...")
                
                # If it's the first pass, we allow single-image posts (num == None) or first image of gallery (num == 1)
                # For passes 2-5, we specifically only want the Nth image of a gallery (num == img_num)
                if img_num == 1:
                    filter_expr = "extension not in ('mp4', 'webm', 'gif', 'gifv') and (num == 1 or num == None)"
                else:
                    filter_expr = f"extension not in ('mp4', 'webm', 'gif', 'gifv') and num == {img_num}"

                # Build the gallery-dl CLI command with all our configurations
                command = [
                    "gallery-dl",
                    "--cookies-from-browser", "firefox",
                    "--filter", filter_expr,
                    "-o", "reddit:api=rest",   # Force REST API to circumvent block
                    "-D", class_dest_path,     # Set output destination (--directory)
                    "-o", "directory=[]",      # Flatten directory structure so we don't get reddit/subreddit subfolders
                ]
                # Append all target urls as fallbacks to ensure we don't run out of posts
                command.extend(target_urls)
                
                # Execute the command, capturing output in real-time
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                
                for line in process.stdout:
                    # Print output to the console so we see the progress
                    print(line, end="")
                    
                    # Check directory count periodically (when an image is processed)
                    if line.strip().lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                        download_count = len([f for f in os.listdir(class_dest_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
                        
                        if download_count >= 200:
                            print(f"\n[INFO] Reached exactly 200 images for {class_name}. Terminating gallery-dl.")
                            process.terminate()
                            break
                
                process.wait()
                
                # Aggressive cleanup: remove .part files, sqlite journals, or other non-images. Don"t ask me why I had to do this
                all_files = os.listdir(class_dest_path)
                valid_exts = ('.jpg', '.jpeg', '.png', '.webp')
                
                for f in all_files:
                    if not f.lower().endswith(valid_exts):
                        file_path = os.path.join(class_dest_path, f)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                
                final_images = sorted([f for f in os.listdir(class_dest_path) if f.lower().endswith(valid_exts)])
                final_count = len(final_images)
                
                if final_count > 200:
                    print(f"Cleaning up {final_count - 200} extra files...")
                    for extra_file in final_images[200:]:
                        os.remove(os.path.join(class_dest_path, extra_file))
                    final_count = 200
                
                if final_count >= 200:
                    break
                
            print(f"Finished processing {class_name}. Total images: {final_count}")
            
        except FileNotFoundError:
            print("Error: gallery-dl executable not found. Make sure it's installed and in your PATH.")
            break
        except KeyboardInterrupt:
            print("\nInterrupted by user. Exiting...")
            if 'process' in locals() and process.poll() is None:
                process.terminate()
            break

if __name__ == "__main__":
    download_reddit_images()