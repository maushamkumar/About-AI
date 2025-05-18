import datetime

from .calendar_utils import get_calendar_service, parse_datetime


def create_event(
    summary: str, 
    start_time: str, 
    end_time: str, 
) -> dict:
    """
    Create a new event in Google Calendar. 
    
    Args: 
        summary(str): Event title/summary
        start_time (str): Start time (e.g., '2025-04-31 14:20)
        end_time (str): End time (e.g., '2025-04-31 15:20)
        
    Returns: 
        dict: Information about the created event or error details
    """
    try: 
        # Get calendar service 
        service = get_calendar_service()
        if not service:
            return{
                "status": "error", 
                "message": "Failed to authenticate with Google Calendar. Please check credentials"
            }
            
        # Always use primary calendar
        calendar_id = 'primary'
        
        # parse time 
        start_dt = parse_datetime(start_time)
        end_dt = parse_datetime(end_time)
        
        if not start_dt or not end_dt:
            return{
                "status" : "error", 
                "message": "Invalid date/time format. Please use YYYY-MM-DD HH:MM format. "
            }
        # Dynamically datermine timezone 
        timezone_id = "Asia/Kolkata"
        
        try: 
            # Try to get the timezone from the calendar settings 
            settings = service.settings().list().execute()
            for setting in settings.get("items", []):
                if setting.get('id') == 'timezone':
                    timezone_id = setting.get('value')
                    break
                
        except Exception: 
            # If we can't get it from settings, we'll use the default 
            pass
        
        # Create event body without type annotations 
        event_body = {}
        
        # Add summary 
        event_body['summary'] = summary
        
        # Add start and end times with the dynamically determined timezone
        event_body['start'] = {
            'dateTime': start_dt.isoformat(), 
            'timeZone' : timezone_id
        }
        
        event_body['end'] = {
            "dateTime": end_dt.isoformat(), 
            'timeZone': timezone_id
        }

        # call the Calendar API or create the event 
        event = (
            service.events().insert(calendarId = calendar_id, body = event_body).execute()
        )
        
        return {
            'status': "Success", 
            "message": "Event Created successful", 
            'event_id': event['id'], 
            'envent_lin': event.get('htmlLink', "")
        
        }
    except Exception as e: 
        return {
            "status": "error", 
            'message': f"Error creating event: {str(e)}"
        }