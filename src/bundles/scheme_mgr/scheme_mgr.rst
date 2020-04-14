html_scheme
-----------

Qt QWebEngineUrlScheme https://doc.qt.io/qt-5/qwebengineurlscheme.html

Alternative web page schemes can be registered using the manager/provider 
mechanishm in bundle_info.xml.

To register a URL scheme, add a provider in a bundle's bundle_info.xml:

::
   
  <Providers>
    <Provider name="help" manager="url_schemes" syntax="Path"/>
  </Providers>

The required parts are the *name* of scheme and the *manager* **url_schemes**.
There are optional keyword arguments to the provider that give access to
all of the customization options of the QWebEngineUrlScheme class (see its
documentation for details).

        defaultPort: int
                Set default port.

        path: str
                One of **Path**, **Host**, **HostAndPort**, or **HostPortAndUserInformation**.

        flag: boolean
                The flag can be one of **SecureScheme**, **LocalScheme**, **LocalAccessAllowed**,
                **NoAccessAllowed**, **ServiceWorkersAllowed**, **ViewSourceAllowed**,
                or **ContentSecurityPolicyIgnored**.
                The boolean should be **true**, **1**, or **on**.
