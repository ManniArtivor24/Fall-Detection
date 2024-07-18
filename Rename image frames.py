import os

def rename_files_in_folder(folder_path):
    # List all files in the folder
    files = os.listdir(folder_path)

    # Sort the files to ensure consistent naming
    files.sort()

    # Rename each file
    for index, filename in enumerate(files):
        # Get the file extension
        file_extension = os.path.splitext(filename)[1]

        # Create the new filename
        new_filename = f"frame{index + 1:04d}{file_extension}"

        # Full path to the current and new file
        old_file = os.path.join(folder_path, filename)
        new_file = os.path.join(folder_path, new_filename)

        # Rename the file
        os.rename(old_file, new_file)
        print(f"Renamed: {old_file} to {new_file}")


# Replace 'your_folder_path_here' with the path to your folder
rename_files_in_folder('UP Fall Dataset/Falling from seated position - Activity 5 ')
