from functools import wraps


def callback(cache_name):
    def decorator(setter_func):
        @wraps(setter_func)
        def wrapper(self, new_value, *args, **kwargs):
            # Get the current value of the property
            prop_name = setter_func.__name__
            current_value = getattr(self, f"_{prop_name}")

            # Call the original setter function
            result = setter_func(self, new_value, *args, **kwargs)

            # If the value has changed, invalidate the cache
            if current_value != new_value:
                self.__dict__.pop(cache_name, None)

            return result
        
        return wrapper
    
    return decorator
