using Microsoft.AspNetCore.Http;

namespace TTC.Siren.Server.Security;

/// <summary>How a request proved who it is: by an API key in a header (the plugin, or any program), or by the session cookie of the web login.</summary>
public enum AuthSource
{
    ApiKey,
    Session,
}

/// <summary>The outcome of authenticating a request: the key it belongs to, or the HTTP status and error to answer.</summary>
public readonly record struct AuthResult(bool Ok, string KeyId, int Status, string? Error, AuthSource Source = AuthSource.ApiKey)
{
    public static AuthResult Allowed(string keyId) => new(true, keyId, 200, null);
    public static AuthResult Denied(int status, string error) => new(false, "", status, error);
    public static AuthResult FromSession(string keyId) => new(true, keyId, 200, null, AuthSource.Session);
}

/// <summary>Decides which API key a request belongs to. The game logs and the live stream are scoped by the key id it returns (never by the secret).</summary>
public interface IApiKeyAuthenticator
{
    AuthResult Authenticate(HttpRequest request);
}

/// <summary>Development default: no authentication, every request belongs to the key "local". Replaced by the key store in the cloud setup.</summary>
public sealed class OpenAuthenticator : IApiKeyAuthenticator
{
    public AuthResult Authenticate(HttpRequest request) => AuthResult.Allowed("local");
}
