using System.IO.Compression;
using System.Text.Json;

namespace TTC.Siren.Tests;

/// <summary>
/// Phase 8A golden data (tools/phase8a): the CPython random sequences, the parsed and sampled requests, and the end-to-end responses of
/// the Python server. Same rules as <see cref="GoldenFiles"/>: generated on demand by the Python generator, bound to the Python source
/// by the SHA-256 header (the file sets mirror tools/phase8a/golden_sources8.py), regenerated when stale, never skipped.
/// </summary>
internal static class Golden8
{
    public const int RngSeed = 20261011;
    public const int RngCount = 600;
    public const string RngGeneratorVersion = "p8_rng/1";

    private static readonly object Gate = new();

    private static readonly string[] RngPatterns =
    {
        "tools/phase8a/golden_sources8.py", "tools/phase8a/py_rng_trace.py",
    };

    private static readonly string[] FmtPatterns = { "tools/phase8a/golden_sources8.py", "tools/phase8a/py_format_trace.py" };

    private static readonly string[] ParsePatterns =
    {
        "core/*.py", "csv_compat.py", "data/幻卡数据库.csv", "ai_server.py", "ai/unknown_card_handler.py", "config/unknown_card_config.py",
        "tools/parity/py_rules_trace.py", "tools/phase7/gen_corpus.py", "tools/phase8a/golden_sources8.py", "tools/phase8a/gen_requests.py",
        "tools/phase8a/py_parse_trace.py",
    };

    private static readonly string[] SamplingPatterns =
    {
        "core/*.py", "csv_compat.py", "data/幻卡数据库.csv", "ai_server.py", "ai/unknown_card_handler.py", "config/unknown_card_config.py",
        "tools/parity/py_rules_trace.py", "tools/phase8a/golden_sources8.py", "tools/phase8a/py_sampling_trace.py",
    };

    private static readonly string[] E2ePatterns = ParsePatterns.Concat(new[]
    {
        "ai/ai.py", "ai/csharp_engine.py", "tools/phase8a/py_e2e_trace.py",
    }).ToArray();

    public static string RepoRoot => GoldenFiles.RepoRoot;
    public static string GoldenDir => GoldenFiles.GoldenDir;

    private static string[] Patterns(string kind) => kind switch
    {
        "rng" => RngPatterns,
        "fmt" => FmtPatterns,
        "parse" => ParsePatterns,
        "sampling" => SamplingPatterns,
        "e2e" => E2ePatterns,
        _ => throw new ArgumentException("unknown golden kind " + kind),
    };

    public static string EnsureRng()
    {
        string path = Path.Combine(GoldenDir, $"p8_rng_g1_s{RngSeed}_n{RngCount}.jsonl.gz");
        Ensure(path, "rng", "tools/phase8a/py_rng_trace.py", "--out", "{0}", "--seed", RngSeed.ToString(), "--count", RngCount.ToString());
        return path;
    }

    /// <summary>
    /// Generates (or reuses) a golden file produced by <paramref name="script"/> with <paramref name="args"/>; the placeholder
    /// <c>{0}</c> in the arguments is replaced by a temporary output path that becomes <paramref name="path"/> on success.
    /// </summary>
    public static string Ensure(string path, string kind, string script, params string[] args)
    {
        lock (Gate)
        {
            if (File.Exists(path))
            {
                string? stale = StaleReason(path, kind);
                if (stale is null) return path;
                Console.Error.WriteLine($"golden {Path.GetFileName(path)} is stale ({stale}); regenerating");
                File.Delete(path);
            }
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            string tmp = path + ".tmp";
            if (File.Exists(tmp)) File.Delete(tmp);
            var arguments = new List<string> { script };
            foreach (string a in args) arguments.Add(a == "{0}" ? tmp : a);
            var (exitCode, stdout, stderr) = GoldenFiles.RunPythonProcess(arguments);
            if (exitCode != 0 || !File.Exists(tmp))
                throw new InvalidOperationException($"Golden generation failed (exit {exitCode}):\n{stdout}\n{stderr}");
            File.Move(tmp, path);
            return path;
        }
    }

    public static string? StaleReason(string path, string kind)
    {
        JsonElement header;
        using (var file = File.OpenRead(path))
        {
            if (path.EndsWith(".gz", StringComparison.Ordinal))
            {
                using var gz = new GZipStream(file, CompressionMode.Decompress);
                using var reader = new StreamReader(gz);
                using var doc = JsonDocument.Parse(reader.ReadLine()!);
                header = doc.RootElement.GetProperty("header").Clone();
            }
            else
            {
                using var doc = JsonDocument.Parse(file);
                if (!doc.RootElement.TryGetProperty("header", out var h)) return "no header (golden predates source binding)";
                header = h.Clone();
            }
        }
        return GoldenSources.Check(header, RepoRoot, Patterns(kind), GoldenSources.PythonVersion);
    }

    /// <summary>Reads every line of a gzip JSON-lines golden file after the header line.</summary>
    public static IEnumerable<JsonDocument> ReadRecords(string path)
    {
        using var file = File.OpenRead(path);
        using var gz = new GZipStream(file, CompressionMode.Decompress);
        using var reader = new StreamReader(gz);
        reader.ReadLine(); // header
        string? line;
        while ((line = reader.ReadLine()) is not null)
            yield return JsonDocument.Parse(line);
    }
}
