benchmark_name=benchmark200
project_path=/mnt/storage_4/home/ignacys/pl0134-01/project_data/admire-project
data_folder_name=test_mar_feb_top200
eagle_config_key=eagle

echo "------------------------"
echo "Will be copying ${benchmark_name}"
echo "------------------------"

# Copy the data from remote to local
echo "Copying data from remote to local"
from=$project_path/data/processed/$data_folder_name/
to=./data/processed/$data_folder_name/
rsync -avz --progress $eagle_config_key:$from $to
echo "Done copying data from remote to local"

echo "------------------------"

# Copy the $benchmark_name data from remote to local
echo "Copying $benchmark_name data from remote to local"
from=$project_path/lightning_logs/$benchmark_name/
to=./lightning_logs/$benchmark_name/
rsync -avz --progress $eagle_config_key:$from $to
echo "Done copying $benchmark_name data from remote to local"

echo "------------------------"

# Now we need to create a symlink to the data in proper folders
echo "Creating symlinks to data in proper folders"
ln -s $PWD/lightning_logs/$benchmark_name/${benchmark_name}_CNN $PWD/lightning_logs/AE_CNN/$benchmark_name
ln -s $PWD/lightning_logs/$benchmark_name/${benchmark_name}_LSTMCNN $PWD/lightning_logs/AE_LSTMCNN/$benchmark_name
ln -s $PWD/lightning_logs/$benchmark_name/${benchmark_name}_LSTMPLAIN $PWD/lightning_logs/AE_LSTMPLAIN/$benchmark_name

echo "------------------------"

# Lastly, we need to copy the config file to the root folder
echo "Copying config file to root folder"
cat $PWD/lightning_logs/$benchmark_name/config.ini > $PWD/config.ini

echo "------------------------"
echo "Done copying $benchmark_name"
echo "------------------------"