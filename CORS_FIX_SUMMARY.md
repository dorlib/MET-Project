# CORS Issue Fix Summary

## Problem
When accessing the profile page at `http://localhost:3000/profile`, users encountered a CORS error:
```
Access to XMLHttpRequest at 'http://localhost:8000/user/settings' from origin 'http://localhost:3000' 
has been blocked by CORS policy: Response to preflight request doesn't pass access control check: 
It does not have HTTP ok status.
```

## Root Cause
1. **Missing API Route**: The API gateway was missing the `/user/settings` route, causing 404 errors
2. **Basic CORS Configuration**: CORS was configured with default settings without explicit origins

## Fixes Applied

### 1. Added Missing Route to API Gateway
**File**: `backend/api_gateway/api.py`

Added the missing `/user/settings` route:
```python
@app.route('/user/settings', methods=['GET'])
@token_required
def get_user_settings():
    """
    Get authenticated user's settings including 2FA status
    """
    try:
        # Forward to user service with token
        auth_header = request.headers.get('Authorization')
        
        response = requests.get(
            f"{USER_SERVICE_URL}/user/settings",
            headers={"Authorization": auth_header}
        )
        return response.json(), response.status_code
    except Exception as e:
        logging.error(f"Error fetching user settings: {str(e)}")
        return jsonify({"error": str(e)}), 500
```

### 2. Enhanced CORS Configuration
**Files**: 
- `backend/api_gateway/api.py`
- `backend/user_service/app.py`

Updated CORS configuration to be more explicit:
```python
# Configure CORS with explicit settings
CORS(app, origins=['http://localhost:3000', 'http://127.0.0.1:3000'], 
     methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
     allow_headers=['Content-Type', 'Authorization'],
     supports_credentials=True)
```

### 3. Verification
- ✅ Route `/user/settings` now exists in API gateway
- ✅ CORS preflight requests return proper headers:
  - `Access-Control-Allow-Origin: http://localhost:3000`
  - `Access-Control-Allow-Credentials: true`  
  - `Access-Control-Allow-Headers: Authorization`
- ✅ Services rebuilt and restarted successfully

## Testing
```bash
# Test CORS preflight
Invoke-WebRequest -Uri "http://localhost:8000/user/settings" -Method OPTIONS \
  -Headers @{"Origin"="http://localhost:3000"; "Access-Control-Request-Method"="GET"; "Access-Control-Request-Headers"="Authorization"}

# Returns 200 OK with proper CORS headers
```

## Result
The profile page should now load without CORS errors and be able to fetch user settings from the backend API.

## Additional Fix: Scan Deletion Issue

### Problem
When trying to delete a scan from the profile page, users encountered another CORS error:
```
Access to XMLHttpRequest at 'http://localhost:8000/user/scans/{scan_id}' from origin 'http://localhost:3000' 
has been blocked by CORS policy: Response to preflight request doesn't pass access control check: 
It does not have HTTP ok status.
```

### Root Cause
The API gateway was missing the DELETE route for `/user/scans/<job_id>`, even though the user service had the corresponding endpoint.

### Fix Applied
**File**: `backend/api_gateway/api.py`

Added the missing DELETE route for individual scan deletion:
```python
@app.route('/user/scans/<job_id>', methods=['DELETE'])
@token_required
def delete_user_scan(job_id):
    """
    Delete a specific scan for the authenticated user
    """
    try:
        # Forward to user service with token
        auth_header = request.headers.get('Authorization')
        
        response = requests.delete(
            f"{USER_SERVICE_URL}/scans/{job_id}",
            headers={"Authorization": auth_header}
        )
        return response.json(), response.status_code
    except Exception as e:
        logging.error(f"Error deleting scan: {str(e)}")
        return jsonify({"error": str(e)}), 500
```

### Verification
- ✅ DELETE route `/user/scans/<job_id>` now exists in API gateway
- ✅ CORS preflight for DELETE requests returns proper headers
- ✅ Route properly forwards requests to user service

### Result
Users can now successfully delete scans from their profile page without CORS errors.

## Database Connection Pool Fix

### Problem
After deleting all scans and refreshing the profile page, users encountered:
```
GET http://localhost:8000/user/scans?page=1&per_page=5 500 (INTERNAL SERVER ERROR)
GET http://localhost:8000/user/settings 401 (UNAUTHORIZED)
```

The logs showed database connection pool timeout errors:
```
sqlalchemy.exc.TimeoutError: QueuePool limit of size 5 overflow 10 reached, connection timed out, timeout 30.00
```

### Root Cause
Multiple endpoints in the user service were using `db = next(get_db())` without properly closing database connections. This exhausted the connection pool over time, especially after multiple operations like scan deletions.

### Fixes Applied
**Files**: 
- `backend/user_service/controllers/controllers.py`
- `backend/user_service/app.py`

Fixed database session management in critical endpoints:

1. **GET /scans endpoint** - Fixed scan listing with proper connection closure
2. **DELETE /scans/<job_id> endpoint** - Fixed scan deletion with proper connection closure  
3. **GET /user endpoint** - Fixed user profile retrieval (used for token verification)
4. **GET /user/settings endpoint** - Fixed user settings retrieval

**Pattern Applied:**
```python
# Use database session as context manager
db_gen = get_db()
db = next(db_gen)
try:
    # Database operations here
    result = SomeService.operation(db, ...)
    return jsonify(result)
finally:
    # Properly close the database session
    db.close()
```

### Verification
- ✅ Profile page loads without 500 errors
- ✅ User settings endpoint works correctly
- ✅ Scan listing works after deletions
- ✅ Database connection pool no longer exhausted
- ✅ All CORS issues resolved

### Result
The profile page now works correctly without database connection errors, and users can navigate and manage their scans without issues.
