#  Readme dla skryptów w folderze models

##  Opis
### ae_dataloader.py
Klasa dziedzicząca po Torchowym 'Dataset' do wczytywania odpowiednio przygotowanych wcześniej danych z folderu data/preprocessing. 
Dane wczytywane są w formacie .parquet dzięki czemu jest to bardzo wydajne rozwiązanie (space & time).
Dane są konkatenowane w jeden numpy array o kształcie `(n_channels, n_nodes, n_time_awate_measuerements)`.
Channels to kolejne kanały, które są w tym przypadku: `'power', 'cpu1', 'cpu2'`.
Nodes to kolejne węzły, a time_awate_measuerements to kolejne pomiaru zmiennej z channel osadzone w kolejności zgodnej z czasem (każdy węzeł i każdy channel jest tej samej długości). 

Nadpisane są dwie metody z klasy Dataset:  
    - `\__len__` - zwraca długość danych -> długość danych obliczana jest na podstawie szerokości przesuwanego okienka, które jest przesuwane po danych  
    - `\__getitem__` - zwraca pojedynczy element danych -> element danych jest zwracany w postaci numpy array o kształcie (n_channels, n_nodes, n_time_awate_measuerements), gdzie szerokość n_time_awate_measuerements jest zależne od parametru 'window_size' (domyślnie 20)  

### ae_litmodel.py
Klasa dziedzicząca po pytorch_lightning.LightningModule. 

### ae_train.py
Skrypt do trenowania modelu.