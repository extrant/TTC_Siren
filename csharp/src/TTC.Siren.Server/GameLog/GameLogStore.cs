using System.Collections.Concurrent;
using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace TTC.Siren.Server.GameLog;

/// <summary>Settings of the game logs (<c>Server:GameLog</c>).</summary>
public sealed class GameLogOptions
{
    /// <summary>Log games at all. Off: nothing is written, the API lists nothing.</summary>
    public bool Enabled { get; set; } = true;

    /// <summary>Directory of the logs; one subdirectory per API key, one JSONL file per game. Never commit it: it holds users' requests.</summary>
    public string Directory { get; set; } = Path.Combine(AppContext.BaseDirectory, "gamelogs");

    /// <summary>Files not written for this many days are deleted by the cleanup job.</summary>
    public int RetentionDays { get; set; } = 30;

    /// <summary>A request continues a game only if it arrives within this many minutes of the previous one (game identity by inference).</summary>
    public int GapMinutes { get; set; } = 15;

    public void Validate()
    {
        if (RetentionDays < 1) throw new ArgumentException("GameLog:RetentionDays must be at least 1");
        if (GapMinutes < 1) throw new ArgumentException("GameLog:GapMinutes must be at least 1");
    }
}

/// <summary>One line of the games list.</summary>
public sealed record GameSummary(string GameId, DateTimeOffset Started, DateTimeOffset LastWrite, int Records, int Requests, bool Ended, string? Winner,
    string? MyOwner, string? Rules, string? IdSource);

/// <summary>
/// File storage of the game logs: <c>&lt;directory&gt;/&lt;key id&gt;/&lt;game id&gt;.jsonl</c>, append-only, one JSON record per line (schema in
/// docs/design/GAME_LOG_SCHEMA.md). Key and game ids are validated before they touch a path, so a request cannot reach another directory. A game
/// belongs to the key that wrote it; every operation takes the key id, so a key can only list, read and delete its own games.
/// </summary>
public sealed class GameLogStore
{
    public static readonly Regex GameIdPattern = new("^[A-Za-z0-9_-]{1,64}$", RegexOptions.Compiled);
    private static readonly Regex KeyIdPattern = new("^[A-Za-z0-9_-]{1,64}$", RegexOptions.Compiled);

    private readonly GameLogOptions _options;
    private readonly TimeProvider _time;
    private readonly ConcurrentDictionary<string, SemaphoreSlim> _gates = new();

    public GameLogStore(GameLogOptions options, TimeProvider? time = null)
    {
        options.Validate();
        _options = options;
        _time = time ?? TimeProvider.System;
    }

    public GameLogOptions Options => _options;

    private string KeyDirectory(string keyId)
    {
        if (!KeyIdPattern.IsMatch(keyId)) throw new ArgumentException("invalid key id", nameof(keyId));
        return Path.Combine(_options.Directory, keyId);
    }

    private string PathOf(string keyId, string gameId)
    {
        if (!GameIdPattern.IsMatch(gameId)) throw new ArgumentException("invalid game id", nameof(gameId));
        return Path.Combine(KeyDirectory(keyId), gameId + ".jsonl");
    }

    private SemaphoreSlim Gate(string keyId, string gameId) => _gates.GetOrAdd(keyId + "/" + gameId, _ => new SemaphoreSlim(1, 1));

    public bool Exists(string keyId, string gameId) => File.Exists(PathOf(keyId, gameId));

    /// <summary>Appends one record (a single JSON line) to the game's file, creating it and the key's directory on first use.</summary>
    public void Append(string keyId, string gameId, string jsonLine)
    {
        string path = PathOf(keyId, gameId);
        var gate = Gate(keyId, gameId);
        gate.Wait();
        try
        {
            System.IO.Directory.CreateDirectory(KeyDirectory(keyId));
            using var stream = new FileStream(path, FileMode.Append, FileAccess.Write, FileShare.ReadWrite | FileShare.Delete);
            var bytes = Encoding.UTF8.GetBytes(jsonLine + "\n");
            stream.Write(bytes, 0, bytes.Length);
            stream.Flush(flushToDisk: true);
        }
        finally
        {
            gate.Release();
        }
    }

    /// <summary>All lines of a game (empty when the game does not exist for this key).</summary>
    public List<string> ReadLines(string keyId, string gameId)
    {
        string path = PathOf(keyId, gameId);
        if (!File.Exists(path)) return new List<string>();
        var gate = Gate(keyId, gameId);
        gate.Wait();
        try
        {
            using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);
            using var reader = new StreamReader(stream, Encoding.UTF8);
            var lines = new List<string>();
            string? line;
            while ((line = reader.ReadLine()) is not null)
                if (line.Length > 0) lines.Add(line);
            return lines;
        }
        finally
        {
            gate.Release();
        }
    }

    public bool Delete(string keyId, string gameId)
    {
        string path = PathOf(keyId, gameId);
        var gate = Gate(keyId, gameId);
        gate.Wait();
        try
        {
            if (!File.Exists(path)) return false;
            File.Delete(path);
            return true;
        }
        finally
        {
            gate.Release();
        }
    }

    /// <summary>Game ids of a key, most recently written first.</summary>
    public List<(string GameId, DateTimeOffset LastWrite)> Files(string keyId)
    {
        string dir = KeyDirectory(keyId);
        if (!System.IO.Directory.Exists(dir)) return new();
        return System.IO.Directory.EnumerateFiles(dir, "*.jsonl")
            .Select(f => (GameId: Path.GetFileNameWithoutExtension(f), LastWrite: new DateTimeOffset(File.GetLastWriteTimeUtc(f), TimeSpan.Zero)))
            .OrderByDescending(x => x.LastWrite).ThenBy(x => x.GameId, StringComparer.Ordinal).ToList();
    }

    /// <summary>A page of the key's games with a summary read from each file's first and last records.</summary>
    public (List<GameSummary> Items, int Total) List(string keyId, int offset, int limit)
    {
        var files = Files(keyId);
        var page = files.Skip(Math.Max(offset, 0)).Take(Math.Clamp(limit, 1, 200));
        var items = new List<GameSummary>();
        foreach (var (gameId, lastWrite) in page)
        {
            var lines = ReadLines(keyId, gameId);
            if (lines.Count == 0) continue;
            items.Add(Summarize(gameId, lastWrite, lines));
        }
        return (items, files.Count);
    }

    public static GameSummary Summarize(string gameId, DateTimeOffset lastWrite, IReadOnlyList<string> lines)
    {
        DateTimeOffset started = lastWrite;
        string? winner = null, myOwner = null, rules = null, idSource = null;
        bool ended = false;
        int requests = 0;
        foreach (var line in lines)
        {
            using var doc = JsonDocument.Parse(line);
            var r = doc.RootElement;
            string type = r.GetProperty("type").GetString()!;
            if (type == "game_start")
            {
                started = DateTimeOffset.Parse(r.GetProperty("t").GetString()!, CultureInfo.InvariantCulture);
                myOwner = r.GetProperty("my_owner").GetString();
                rules = r.GetProperty("rules_text").GetString();
                idSource = r.GetProperty("id_source").GetString();
            }
            else if (type == "request") requests++;
            else if (type == "game_end")
            {
                ended = true;
                winner = r.GetProperty("winner").GetString();
            }
        }
        return new GameSummary(gameId, started, lastWrite, lines.Count, requests, ended, winner, myOwner, rules, idSource);
    }

    /// <summary>
    /// Deletes every game file not written for <see cref="GameLogOptions.RetentionDays"/> days (and key directories left empty). Returns the number
    /// of games removed. Called at start-up and once a day by the hosted cleanup job; tests call it with a clock of their own.
    /// </summary>
    public int Cleanup()
    {
        if (!System.IO.Directory.Exists(_options.Directory)) return 0;
        var cutoff = _time.GetUtcNow() - TimeSpan.FromDays(_options.RetentionDays);
        int removed = 0;
        foreach (var keyDir in System.IO.Directory.EnumerateDirectories(_options.Directory))
        {
            string keyId = Path.GetFileName(keyDir);
            if (!KeyIdPattern.IsMatch(keyId)) continue;
            foreach (var file in System.IO.Directory.EnumerateFiles(keyDir, "*.jsonl"))
            {
                if (new DateTimeOffset(File.GetLastWriteTimeUtc(file), TimeSpan.Zero) >= cutoff) continue;
                string gameId = Path.GetFileNameWithoutExtension(file);
                if (GameIdPattern.IsMatch(gameId) && Delete(keyId, gameId)) removed++;
            }
            if (!System.IO.Directory.EnumerateFileSystemEntries(keyDir).Any()) System.IO.Directory.Delete(keyDir);
        }
        return removed;
    }
}
