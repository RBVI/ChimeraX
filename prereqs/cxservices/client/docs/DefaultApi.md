# cxservices.DefaultApi

All URIs are relative to *http://webservices.rbvi.ucsf.edu/cxservices/api/v1/*

Method | HTTP request | Description
------------- | ------------- | -------------
[**file_get**](DefaultApi.md#file_get) | **GET** /chimerax/files/{job_id}/{file_name} | Return content of job file on server
[**file_post**](DefaultApi.md#file_post) | **POST** /chimerax/files/{job_id}/{file_name} | Upload job file to server
[**files_get**](DefaultApi.md#files_get) | **GET** /chimerax/files/{job_id} | Return job files on server as zip archive
[**files_list**](DefaultApi.md#files_list) | **GET** /chimerax/files_list/{job_id} | Return list of job files on server
[**files_post**](DefaultApi.md#files_post) | **POST** /chimerax/files/{job_id} | Upload zip archive of job files to server
[**job_delete**](DefaultApi.md#job_delete) | **DELETE** /chimerax/job/{job_id} | Delete job on server
[**job_id**](DefaultApi.md#job_id) | **GET** /chimerax/job_id | Return a new job identifier
[**newer_versions**](DefaultApi.md#newer_versions) | **GET** /chimerax/newer | Return list of newer ChimeraX releases (version, URL)
[**sleep**](DefaultApi.md#sleep) | **POST** /chimerax/job/{job_id}/sleep | Sleep for a while and exit
[**status**](DefaultApi.md#status) | **GET** /chimerax/job/{job_id} | Return status of job
[**submit**](DefaultApi.md#submit) | **POST** /chimerax/job/{job_id}/{service_name} | Submit a job for execution

# **file_get**
> str file_get(job_id, file_name)

Return content of job file on server

### Example
```python
from __future__ import print_function
import time
import cxservices
from cxservices.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = cxservices.DefaultApi()
job_id = 'job_id_example' # str | Job identifier
file_name = 'file_name_example' # str | File to fetch

try:
    # Return content of job file on server
    api_response = api_instance.file_get(job_id, file_name)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling DefaultApi->file_get: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **job_id** | **str**| Job identifier | 
 **file_name** | **str**| File to fetch | 

### Return type

**str**

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/octet-stream, application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **file_post**
> file_post(body, job_id, file_name)

Upload job file to server

### Example
```python
from __future__ import print_function
import time
import cxservices
from cxservices.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = cxservices.DefaultApi()
body = cxservices.Object() # Object | Content of file being uploaded
job_id = 'job_id_example' # str | Job identifier
file_name = 'file_name_example' # str | File to fetch

try:
    # Upload job file to server
    api_instance.file_post(body, job_id, file_name)
except ApiException as e:
    print("Exception when calling DefaultApi->file_post: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **body** | **Object**| Content of file being uploaded | 
 **job_id** | **str**| Job identifier | 
 **file_name** | **str**| File to fetch | 

### Return type

void (empty response body)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: application/octet-stream
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **files_get**
> str files_get(job_id)

Return job files on server as zip archive

### Example
```python
from __future__ import print_function
import time
import cxservices
from cxservices.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = cxservices.DefaultApi()
job_id = 'job_id_example' # str | Job identifier

try:
    # Return job files on server as zip archive
    api_response = api_instance.files_get(job_id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling DefaultApi->files_get: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **job_id** | **str**| Job identifier | 

### Return type

**str**

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/zip, application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **files_list**
> InlineResponse200 files_list(job_id)

Return list of job files on server

### Example
```python
from __future__ import print_function
import time
import cxservices
from cxservices.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = cxservices.DefaultApi()
job_id = 'job_id_example' # str | Job identifier

try:
    # Return list of job files on server
    api_response = api_instance.files_list(job_id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling DefaultApi->files_list: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **job_id** | **str**| Job identifier | 

### Return type

[**InlineResponse200**](InlineResponse200.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **files_post**
> files_post(body, job_id)

Upload zip archive of job files to server

### Example
```python
from __future__ import print_function
import time
import cxservices
from cxservices.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = cxservices.DefaultApi()
body = cxservices.Object() # Object | Zip archive of files being uploaded
job_id = 'job_id_example' # str | Job identifier

try:
    # Upload zip archive of job files to server
    api_instance.files_post(body, job_id)
except ApiException as e:
    print("Exception when calling DefaultApi->files_post: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **body** | **Object**| Zip archive of files being uploaded | 
 **job_id** | **str**| Job identifier | 

### Return type

void (empty response body)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: application/zip
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **job_delete**
> job_delete(job_id)

Delete job on server

### Example
```python
from __future__ import print_function
import time
import cxservices
from cxservices.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = cxservices.DefaultApi()
job_id = 'job_id_example' # str | Job identifier

try:
    # Delete job on server
    api_instance.job_delete(job_id)
except ApiException as e:
    print("Exception when calling DefaultApi->job_delete: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **job_id** | **str**| Job identifier | 

### Return type

void (empty response body)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **job_id**
> InlineResponse201 job_id()

Return a new job identifier

### Example
```python
from __future__ import print_function
import time
import cxservices
from cxservices.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = cxservices.DefaultApi()

try:
    # Return a new job identifier
    api_response = api_instance.job_id()
    pprint(api_response)
except ApiException as e:
    print("Exception when calling DefaultApi->job_id: %s\n" % e)
```

### Parameters
This endpoint does not need any parameter.

### Return type

[**InlineResponse201**](InlineResponse201.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **newer_versions**
> list[list[str]] newer_versions(os, os_version, chimera_x_version)

Return list of newer ChimeraX releases (version, URL)

### Example
```python
from __future__ import print_function
import time
import cxservices
from cxservices.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = cxservices.DefaultApi()
os = 'os_example' # str | operating system name
os_version = 'os_version_example' # str | operating system version
chimera_x_version = 'chimera_x_version_example' # str | ChimeraX version to compare with

try:
    # Return list of newer ChimeraX releases (version, URL)
    api_response = api_instance.newer_versions(os, os_version, chimera_x_version)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling DefaultApi->newer_versions: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **os** | **str**| operating system name | 
 **os_version** | **str**| operating system version | 
 **chimera_x_version** | **str**| ChimeraX version to compare with | 

### Return type

**list[list[str]]**

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **sleep**
> sleep(job_id, body=body)

Sleep for a while and exit

### Example
```python
from __future__ import print_function
import time
import cxservices
from cxservices.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = cxservices.DefaultApi()
job_id = 'job_id_example' # str | Job identifier
body = NULL # object | Length of time to sleep (optional)

try:
    # Sleep for a while and exit
    api_instance.sleep(job_id, body=body)
except ApiException as e:
    print("Exception when calling DefaultApi->sleep: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **job_id** | **str**| Job identifier | 
 **body** | [**object**](object.md)| Length of time to sleep | [optional] 

### Return type

void (empty response body)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **status**
> InlineResponse2001 status(job_id)

Return status of job

### Example
```python
from __future__ import print_function
import time
import cxservices
from cxservices.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = cxservices.DefaultApi()
job_id = 'job_id_example' # str | Job identifier

try:
    # Return status of job
    api_response = api_instance.status(job_id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling DefaultApi->status: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **job_id** | **str**| Job identifier | 

### Return type

[**InlineResponse2001**](InlineResponse2001.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/octet-stream, application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **submit**
> submit(body, job_id, service_name)

Submit a job for execution

### Example
```python
from __future__ import print_function
import time
import cxservices
from cxservices.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = cxservices.DefaultApi()
body = NULL # dict(str, object) | Parameters for job execution
job_id = 'job_id_example' # str | Job identifier
service_name = 'service_name_example' # str | Service to invoke

try:
    # Submit a job for execution
    api_instance.submit(body, job_id, service_name)
except ApiException as e:
    print("Exception when calling DefaultApi->submit: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **body** | [**dict(str, object)**](dict.md)| Parameters for job execution | 
 **job_id** | **str**| Job identifier | 
 **service_name** | **str**| Service to invoke | 

### Return type

void (empty response body)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

