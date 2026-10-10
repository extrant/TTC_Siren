using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace TTC.Siren.Server.Security;

/// <summary>A stored API key. The secret itself is never stored: only its SHA-256 hash. <see cref="Id"/> is the public, non-secret name used in logs and as the owner of game logs.</summary>
public sealed record ApiKeyRecord(string Id, string Name, DateTimeOffset Created, bool Enabled, string Hash, int RatePerMinute, int Burst, int MaxInFlight);

/// <summary>
/// The store of API keys (decision A-21): random 256-bit keys, handed out once by the admin CLI and kept only as SHA-256 hashes in a JSON file.
/// The running server notices a change of the file (the CLI edits it) within a second, so a revoked key stops working without a restart.
/// A key looks like <c>ttc_</c> + 43 characters of base64url; only the first 8 characters ever appear anywhere else (the CLI list shows the hash prefix).
/// </summary>
public sealed class ApiKeyStore
{
    public const string Prefix = "ttc_";
    private readonly string _path;
    private readonly TimeProvider _time;
    private readonly object _lock = new();
    private List<ApiKeyRecord> _records = new();
    private Dictionary<string, ApiKeyRecord> _byHash = new(StringComparer.Ordinal);
    private DateTime _loadedWrite = DateTime.MinValue;
    private long _loadedLength = -1;
    private long _nextCheck;

    public ApiKeyStore(string path, TimeProvider? time = null)
    {
        _path = path;
        _time = time ?? TimeProvider.System;
        Reload(force: true);
    }

    public string Path => _path;

    public static string HashOf(string secret) => Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(secret))).ToLowerInvariant();

    private void Reload(bool force)
    {
        lock (_lock)
        {
            if (!File.Exists(_path))
            {
                _records = new();
                _byHash = new(StringComparer.Ordinal);
                _loadedWrite = DateTime.MinValue;
                _loadedLength = -1;
                return;
            }
            var info = new FileInfo(_path);
            if (!force && info.LastWriteTimeUtc == _loadedWrite && info.Length == _loadedLength) return;
            using var stream = new FileStream(_path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);
            using var doc = JsonDocument.Parse(stream);
            var list = new List<ApiKeyRecord>();
            foreach (var k in doc.RootElement.GetProperty("keys").EnumerateArray())
                list.Add(new ApiKeyRecord(k.GetProperty("id").GetString()!, k.GetProperty("name").GetString()!,
                    DateTimeOffset.Parse(k.GetProperty("created").GetString()!, System.Globalization.CultureInfo.InvariantCulture), k.GetProperty("enabled").GetBoolean(),
                    k.GetProperty("hash").GetString()!, k.GetProperty("rate_per_minute").GetInt32(), k.GetProperty("burst").GetInt32(), k.GetProperty("max_in_flight").GetInt32()));
            _records = list;
            _byHash = list.ToDictionary(r => r.Hash, r => r, StringComparer.Ordinal);
            _loadedWrite = info.LastWriteTimeUtc;
            _loadedLength = info.Length;
        }
    }

    private void Save(List<ApiKeyRecord> records)
    {
        var dir = System.IO.Path.GetDirectoryName(System.IO.Path.GetFullPath(_path))!;
        Directory.CreateDirectory(dir);
        var json = JsonSerializer.Serialize(new
        {
            version = 1,
            keys = records.Select(r => new
            {
                id = r.Id, name = r.Name, created = r.Created.UtcDateTime.ToString("yyyy-MM-ddTHH:mm:ssZ", System.Globalization.CultureInfo.InvariantCulture), enabled = r.Enabled,
                hash = r.Hash, rate_per_minute = r.RatePerMinute, burst = r.Burst, max_in_flight = r.MaxInFlight,
            }),
        }, new JsonSerializerOptions { WriteIndented = true });
        string temp = _path + "." + Guid.NewGuid().ToString("N") + ".tmp";
        File.WriteAllText(temp, json, new UTF8Encoding(false));
        File.Move(temp, _path, overwrite: true); // atomic replace: a reader never sees half a file
        var info = new FileInfo(_path);
        _loadedWrite = info.LastWriteTimeUtc;
        _loadedLength = info.Length;
    }

    /// <summary>Re-reads the file now (the operator endpoint calls this after the CLI changed it).</summary>
    public void ReloadNow() => Reload(force: true);

    public IReadOnlyList<ApiKeyRecord> List()
    {
        Reload(force: false);
        lock (_lock) return _records.ToList();
    }

    /// <summary>Creates a key. Returns the record and the SECRET, which is shown once and cannot be recovered.</summary>
    public (ApiKeyRecord Record, string Secret) Add(string name, int ratePerMinute, int burst, int maxInFlight)
    {
        if (string.IsNullOrWhiteSpace(name) || name.Length > 80) throw new ArgumentException("a key needs a name of 1 to 80 characters");
        lock (_lock)
        {
            Reload(force: false);
            if (_records.Any(r => string.Equals(r.Name, name, StringComparison.OrdinalIgnoreCase))) throw new InvalidOperationException($"a key named '{name}' already exists");
            string secret = Prefix + Convert.ToBase64String(RandomNumberGenerator.GetBytes(32)).TrimEnd('=').Replace('+', '-').Replace('/', '_');
            string id;
            do id = "k-" + Convert.ToHexString(RandomNumberGenerator.GetBytes(4)).ToLowerInvariant();
            while (_records.Any(r => r.Id == id));
            var record = new ApiKeyRecord(id, name, _time.GetUtcNow(), true, HashOf(secret), ratePerMinute, burst, maxInFlight);
            var list = _records.Append(record).ToList();
            Save(list);
            _records = list;
            _byHash = list.ToDictionary(r => r.Hash, r => r, StringComparer.Ordinal);
            return (record, secret);
        }
    }

    private ApiKeyRecord? Locate(string idOrName) =>
        _records.FirstOrDefault(r => r.Id == idOrName) ?? _records.FirstOrDefault(r => string.Equals(r.Name, idOrName, StringComparison.OrdinalIgnoreCase));

    public bool Revoke(string idOrName)
    {
        lock (_lock)
        {
            Reload(force: false);
            var found = Locate(idOrName);
            if (found is null) return false;
            return Replace(found, found with { Enabled = false });
        }
    }

    public bool SetLimit(string idOrName, int? ratePerMinute, int? burst, int? maxInFlight)
    {
        lock (_lock)
        {
            Reload(force: false);
            var found = Locate(idOrName);
            if (found is null) return false;
            return Replace(found, found with
            {
                RatePerMinute = ratePerMinute ?? found.RatePerMinute, Burst = burst ?? found.Burst, MaxInFlight = maxInFlight ?? found.MaxInFlight,
            });
        }
    }

    private bool Replace(ApiKeyRecord old, ApiKeyRecord updated)
    {
        var list = _records.Select(r => r.Id == old.Id ? updated : r).ToList();
        Save(list);
        _records = list;
        _byHash = list.ToDictionary(r => r.Hash, r => r, StringComparer.Ordinal);
        return true;
    }

    public ApiKeyRecord? GetById(string id)
    {
        CheckFile();
        lock (_lock) return _records.FirstOrDefault(r => r.Id == id);
    }

    /// <summary>The record of a presented secret (enabled or not), or null. The file is re-read when it changed (checked at most once a second).</summary>
    public ApiKeyRecord? Find(string secret)
    {
        CheckFile();
        string hash = HashOf(secret);
        lock (_lock) return _byHash.TryGetValue(hash, out var r) ? r : null;
    }

    private void CheckFile()
    {
        long now = _time.GetUtcNow().UtcTicks; // the clock of the provider (tests move it by hand)
        if (now < Interlocked.Read(ref _nextCheck)) return;
        Interlocked.Exchange(ref _nextCheck, now + TimeSpan.TicksPerSecond); // once a second
        try
        {
            Reload(force: false);
        }
        catch (Exception e) when (e is IOException or JsonException)
        {
            // keep the keys we have: a half-written file is retried at the next check
        }
    }
}
