from datetime import datetime

class Anomaly:
    id: int
    description: str
    severity: int
    date_start: datetime
    date_end: datetime
    tags: list[str]
    
    def __init__(self, id: int, 
                 description: str, 
                 severity: int, 
                 date_start: datetime | str, 
                 date_end: datetime | str, 
                 tags: list[str],
                 tz: datetime.tzinfo = None
                 ) -> None:
        self.id = id
        self.description = description
        self.severity = severity
        self.tags = tags
        
        if isinstance(date_start, str):
            date_start = datetime.strptime(date_start, '%Y-%m-%d %H:%M:%S')
            if tz is not None:
                date_start = date_start.replace(tzinfo=tz)
            self.date_start = date_start
        if isinstance(date_end, str):
            date_end = datetime.strptime(date_end, '%Y-%m-%d %H:%M:%S')
            if tz is not None:
                date_end = date_end.replace(tzinfo=tz)
            self.date_end = date_end
            
if __name__ == '__main__':
    # Example
    tzone = datetime.now().astimezone().tzinfo
    anomaly = Anomaly(1, 'Test anomaly', 1, '2021-01-01 00:00:00', '2021-01-01 01:00:00', ['test'], tz=tzone)
    
    print(anomaly.date_start)