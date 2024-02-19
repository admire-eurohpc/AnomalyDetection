# Opis stworzony jest z danych ogólnych i step-1 po przefiltrowaniu do jobów dłuższych niż 5 minut i z niezerowym pomiarem zużywanej średniej energii

1. **steps-tres-consumed-total**
    - zakładamy że jest to miara średniego zużycia w Watach [W]
    - wartości wachają się głównie od 100 do 500
    - ~100 z 6k instancji ma pomiary większe od 500 (maksymalne zużycie jakie teoretycznie może wystąpić według Radka to ~560W), my zakładamy że jest możliwe, że pomiar jest sumą dla nodeów z danego joba
1. **steps-tres-consumed-average**
    - bardzo podobne do total, różnica taka, że nie ma pomiarów >500W
1. **steps-tres-consumed-max/min**
    - ma kilka wartości poza skalą (ogromne wartości w milionach)
    - poza tym większość pomiarów jest równa dla max-min, dla niektórych jednak są różnice
1. **steps-statistics-consumed-energy**
    - miara całościowego zużycia energii przez danego joba, wysoko koreluje z czasem wykonania joba
    - niestety czasami pojawiają się zera pomimo przefiltrowania atrybutu średniej energii dla > 0
    - jednostka prawdobodobie w dżulach
1. **allocation_nodes** i **tres-allocated-node**
    - obie miary zawierają ilość zaalokowanych węzłów dla danego joba
    - niestety część rekordów ma oba atrybuty zupełnie inne
    - tres-allocated-node wydaje się być bardziej sensowny z danych, bo np. ilość zaalokowanych cpu nie może pasować często do allocation_nodes - przykład: allocation_node = 1, cpus = 96 - taka konfiguracja węzła nie istnieje  
1. **required-CPUs** i **tres-allocated-cpu**
    - obie miary podają ilość zaalokowanych CPU ale są dość mocno ortogonalne na co nie mamy wytłumaczenia
    - zazwyczaj się zgadzają tylko w przypadku gdy oba pola raportują 28 cpu
    - często jedno pole raportuje np. 96 a drugie 28  
1. **required-memory** i **tres-allocated-mem**  
    - obie miary podają liczbę zaalokowanej pamięci w MB
    - dużo wartości około 100GB
    - podobnie jak w cpu - liczby w obu polach nie odpowiadają sobie w wielu przypadkach
1. **user**
    - nazwa użytkownika
1. **partition**
    - nazwa partycji na której wystąpiła alokacja
    - najczęściej standard lub plgrid, czasami altair, tesla
1. **group**
    - nazwa grupy użytkownika
    - typowo user lub plgrid-user
1. **priority**
    - int pokazujący priority joba
    - ponoć jest skomplikowany wzór na obliczanie priority, bo wartości mają duży rozstrzał - np. 1,2,1387
1. **steps-time-elapsed**
    - czas egzekucji joba
    - zakładamy że jest podany w sekundach
1. **steps-time-start/end**
    - początek / koniec joba w formie int do przekonwertowania na datetime
