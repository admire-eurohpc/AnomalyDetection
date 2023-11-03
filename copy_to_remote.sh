username="<YOUR USERNAME GOES HERE>"

# Copy all local folders, one by one to the remote server
remote_server="$username@eagle.man.poznan.pl"
eagle_path="/home/users/$username/pl0134-01/project_data/admire-project"

# Print when starting
echo "Copying files to the remote server: $remote_server:$eagle_path"

# Create a list of folders and files to copy
folders=(data admire notebooks scripts README.md requirements.txt config.ini environment.yml requirements.txt)

# Iterate over the list of folders and files
for folder in "${folders[@]}"
do
    # Print the folder name
    echo "Copying $folder to the remote server"
    
    # Copy the folder to the remote server
    rsync -avz --progress $folder $remote_server:$eagle_path
done

# Print when finished
echo "Finished copying files to the remote server"