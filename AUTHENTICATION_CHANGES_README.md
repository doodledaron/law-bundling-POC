# Authentication & Security Changes README

## Overview

This document outlines the major security and endpoint reorganization changes implemented in the Law Document Processing API. These changes provide robust API key authentication while maintaining a clear separation between production and development interfaces.

## 🔐 What Changed

### 1. API Authentication Implementation
- **New requirement**: All production API endpoints now require API key authentication
- **Header-based**: Uses `X-API-Key` header for authentication
- **Configurable**: API keys managed through environment variables
- **Backward compatible**: If no API keys are configured, authentication is disabled with warnings

### 2. Endpoint Reorganization
- **Production endpoints**: Remain unchanged but now require authentication
- **Development endpoints**: Moved to `/dev` prefix and require no authentication
- **Root redirect**: `/` now redirects to `/dev/` for development access

## 📋 Endpoint Changes Summary

### Production API Endpoints (Authentication Required)
| Endpoint | Status | Authentication |
|----------|--------|----------------|
| `POST /api/upload` | ✅ Unchanged | 🔒 Required |
| `GET /api/job/{job_id}` | ✅ Unchanged | 🔒 Required |
| `GET /health` | ✅ Unchanged | 🔓 Public |

### Development Endpoints (No Authentication)
| Old Endpoint | New Endpoint | Notes |
|--------------|--------------|-------|
| `GET /` | `GET /dev/` | Main upload interface |
| `GET /bulk` | `GET /dev/bulk` | Bulk processing page |
| `GET /results-list` | `GET /dev/results-list` | Results list page |
| `POST /bulk-upload` | `POST /dev/bulk-upload` | Bulk upload endpoint |
| `POST /upload` | `POST /dev/upload` | Single upload (HTML form) |
| `GET /job/{job_id}` | `GET /dev/job/{job_id}` | Job status page |
| `GET /api/results` | `GET /dev/api/results` | Internal results API |
| `GET /api/parallel/status` | `GET /dev/api/parallel/status` | System status |
| `GET /api/containers/status` | `GET /dev/api/containers/status` | Container status |

## 🚀 Setup Instructions

### 1. Configure API Keys

Add the following environment variable to your `.env` file or system environment:

```env
# Required: Comma-separated list of valid API keys
API_KEYS=main-api-key-2024,backup-key-xyz,client-specific-key-abc
```

### 2. Environment Variable Options

```env
# Example configuration
API_KEYS=prod-key-1,prod-key-2,dev-key-temp

# For development (disable authentication)
# API_KEYS=

# Single API key
API_KEYS=single-production-key
```

### 3. Docker Environment

Update your `docker-compose.yml` or deployment configuration:

```yaml
services:
  api:
    environment:
      - API_KEYS=your-production-api-keys-here
      # ... other environment variables
```

## 🔧 Usage Examples

### Client Integration (New)

#### cURL Example
```bash
# Upload document with authentication
curl -X POST \
  -H "X-API-Key: your-api-key-here" \
  -F "file=@document.pdf" \
  http://your-domain.com/api/upload

# Check job status
curl -H "X-API-Key: your-api-key-here" \
  http://your-domain.com/api/job/{job_id}
```

#### Python Example
```python
import requests

# Configure headers
headers = {"X-API-Key": "your-api-key-here"}

# Upload document
with open("document.pdf", "rb") as f:
    response = requests.post(
        "http://your-domain.com/api/upload",
        headers=headers,
        files={"file": f}
    )

# Check status
job_id = response.json()["job_id"]
status = requests.get(
    f"http://your-domain.com/api/job/{job_id}",
    headers=headers
)
```

#### JavaScript Example
```javascript
// Configure headers
const headers = {
    'X-API-Key': 'your-api-key-here'
};

// Upload document
const formData = new FormData();
formData.append('file', fileInput.files[0]);

const uploadResponse = await fetch('/api/upload', {
    method: 'POST',
    headers: headers,
    body: formData
});

// Check status
const jobData = await uploadResponse.json();
const statusResponse = await fetch(`/api/job/${jobData.job_id}`, {
    headers: headers
});
```

## ⚠️ Migration Guide

### For Existing Integrations

1. **Update client code** to include `X-API-Key` header in all requests to `/api/*` endpoints
2. **Obtain API keys** from your system administrator
3. **Test integration** with new authentication requirements
4. **Update monitoring/health checks** (note: `/health` endpoint remains public)

### For Development/Testing

1. **Use development endpoints** at `/dev/*` for internal testing (no authentication required)
2. **Access web interface** at `http://your-domain.com/dev/` (redirected from root)
3. **Use production endpoints** only for production integrations with proper API keys

## 🔍 Error Handling

### Authentication Errors

#### Missing API Key (401)
```json
{
  "detail": {
    "error": "Missing API key",
    "message": "Please provide a valid API key in the 'X-API-Key' header",
    "required_header": "X-API-Key"
  }
}
```

#### Invalid API Key (401)
```json
{
  "detail": {
    "error": "Invalid API key",
    "message": "The provided API key is not valid",
    "required_header": "X-API-Key"
  }
}
```

### Client Error Handling Example

```python
import requests

def make_authenticated_request(url, api_key, **kwargs):
    headers = kwargs.get('headers', {})
    headers['X-API-Key'] = api_key
    kwargs['headers'] = headers
    
    response = requests.request(**kwargs)
    
    if response.status_code == 401:
        error_detail = response.json().get('detail', {})
        raise AuthenticationError(f"API authentication failed: {error_detail.get('message', 'Invalid API key')}")
    
    response.raise_for_status()
    return response

class AuthenticationError(Exception):
    pass
```

## 🛡️ Security Features

### 1. API Key Management
- **Environment-based**: Keys stored in environment variables, not in code
- **Multiple keys**: Support for multiple valid API keys
- **Key rotation**: Easy to add/remove keys by updating environment variables
- **Logging**: Invalid key attempts are logged (with partial key for debugging)

### 2. Endpoint Separation
- **Production isolation**: API endpoints require authentication
- **Development access**: Development endpoints accessible without keys
- **Clear boundaries**: `/api/*` for production, `/dev/*` for development

### 3. Monitoring & Debugging
- **Authentication logging**: Valid and invalid attempts are logged
- **Health checks**: Public health endpoint for monitoring systems
- **Error details**: Clear error messages for troubleshooting

## 📊 Monitoring

### Log Messages to Monitor

```bash
# Valid authentication
INFO: Valid API key used: main-api...

# Invalid attempts
WARNING: Invalid API key attempted: invalid-...

# Configuration issues
WARNING: No API keys configured - authentication disabled
```

### Health Check

The `/health` endpoint remains publicly accessible for monitoring:

```bash
curl http://your-domain.com/health
```

## 🔄 Rollback Plan

If you need to disable authentication temporarily:

1. **Remove or comment out** the `API_KEYS` environment variable
2. **Restart the application**
3. **Monitor logs** for "No API keys configured - authentication disabled" message

The system will continue operating with authentication disabled (backward compatibility mode).

## 📞 Support

### For API Key Issues
- Contact your system administrator to obtain valid API keys
- Check environment variable configuration
- Verify header format: `X-API-Key: your-key-here`

### For Integration Issues
- Review the updated API_DOCUMENTATION.md for complete endpoint details
- Test with development endpoints first (`/dev/*`)
- Check authentication error responses for specific guidance

### For System Administration
- Manage API keys through the `API_KEYS` environment variable
- Monitor authentication logs for security issues
- Use development endpoints for internal tools and debugging

---

## 📝 Change Summary

- ✅ API key authentication implemented for production endpoints
- ✅ Development endpoints moved to `/dev` prefix
- ✅ Backward compatibility maintained (authentication can be disabled)
- ✅ Clear error messages and logging
- ✅ Updated documentation and examples
- ✅ Comprehensive migration guide provided

This implementation provides robust security while maintaining ease of use for both production integrations and development workflows.