from .calendar_utils import get_calendar_service, parse_datetime

def edit_event(
    event_id: str, 
    summary: str, 
    start_time: str, 
    end_time: str, 
) -> dict:
    """
    Edit an existing event in Google Calendar - Change title and/or reschedule.
    
    Args: 
        event_id(str): The ID of the event to edit
        summary(str): New title/summary for the envent(pass empty string to keep unchanged)
        start_time (str): New start time (e.g., '2025-04-31 14:40',  pass empty string to keep unchanged)
        end_time (str): New End time (e.g., '2025-04-31 15:40',  pass empty string to keep unchanged)
        
    Returns: 
        dict: Information about the edited event or error details
    """
    
    try: 
        # Get calendar service 
        service = get_calendar_service()
        if not service:
            return{
                "status": "error", 
                "message": "Failed to authenticate with Google Calendar. Please check credentails."
            }
            
        # Always use primary calendar
        calendar_id = "primary"
        
        # First get the existing event 
        try:
            event = (
                service.event().get(calendarId=calendar_id, eventId = event_id).execute()
            )
            
        except Exception:
            return {
                "status": "error", 
                'message': f"Event with ID {event_id} not found in primary calendar.", 
            }
        
        # Update the event with new values
        if summary:
            event['summary'] = summary
            
        # Get timezone from teh original event
        timezone_id = "Asia/Kolkata" # Default 
        
        if 'start' in event and 'timeZone' in event['start']:
            timezone_id = event['start']['timeZone']
            
        # Update start time if provided 
        if start_time:
            start_dt = parse_datetime(start_time)
            if not start_dt:
                return{
                    'status': "error", 
                    'message': 'Invalid start time format. Please use YYYY-MM-DD HH:DD format. ', 
                }
            event['start'] = {
                'dateTime': start_dt.isoformat(), 
                'timeZone': timezone_id
            }
            
        # Update end time if provided 
        if end_time:
            end_time = parse_datetime(end_time)
            if not end_time:
                return{
                    'status': "error", 
                    'message': 'Invalid start time format. Please use YYYY-MM-DD HH:DD format. ', 
                }
            event['start'] = {
                'dateTime': end_time.isoformat(), 
                'timeZone': timezone_id
            }
            
        # Update the event 
        updated_event = (
            service.events().update(calendarId=calendar_id, eventId = event_id, body=event).execute()
        )
        
        return{
            'status': "SUCCESS", 
            'message': "Event updated successfully", 
            'event_id': updated_event['id'], 
            'event_link': updated_event.get('htmlLink', ''),
        }
    
    except Exception as e:
        return {
            'status': "error", 
            'message': f"Error Updating event: {str(e)}"
        }