using System.Text.Json;
using System.Text.Json.Nodes;

namespace TTC.Siren.Tests;

/// <summary>Review M4-2 (M-1): golden files are bound to the Python source they were generated from.</summary>
public sealed class GoldenBindingTests
{
    private static JsonElement ToElement(JsonObject o) => JsonDocument.Parse(o.ToJsonString()).RootElement;

    private static JsonObject HeaderFor(string root, string kind, string pythonVersion)
    {
        var sources = new JsonObject();
        foreach (string rel in GoldenSources.Required(root, kind)) sources[rel] = GoldenSources.Sha256(root, rel);
        return new JsonObject { ["python_version"] = pythonVersion, ["source_sha256"] = sources };
    }

    /// <summary>A temporary copy of the files a kind is bound to, laid out like the repository.</summary>
    private static string CopyTree(string kind)
    {
        string temp = Path.Combine(Path.GetTempPath(), "ttc_golden_binding_" + Guid.NewGuid().ToString("N"));
        foreach (string rel in GoldenSources.Required(GoldenFiles.RepoRoot, kind))
        {
            string dst = Path.Combine(temp, rel);
            Directory.CreateDirectory(Path.GetDirectoryName(dst)!);
            File.Copy(Path.Combine(GoldenFiles.RepoRoot, rel), dst);
        }
        return temp;
    }

    [Fact]
    public void A_matching_header_passes_and_every_kind_of_change_is_detected()
    {
        foreach (string kind in new[] { "rules", "carddb", "search" })
        {
            string temp = CopyTree(kind);
            try
            {
                const string py = "3.11";
                Assert.Null(GoldenSources.Check(ToElement(HeaderFor(temp, kind, py)), temp, kind, py));

                // The running Python differs from the one that generated the data.
                Assert.Contains("Python", GoldenSources.Check(ToElement(HeaderFor(temp, kind, py)), temp, kind, "3.12")!);

                // Every bound file is checked: changing a single byte of any one of them is detected, and named.
                foreach (string rel in GoldenSources.Required(temp, kind))
                {
                    string path = Path.Combine(temp, rel);
                    byte[] original = File.ReadAllBytes(path);
                    var header = ToElement(HeaderFor(temp, kind, py));
                    File.WriteAllBytes(path, original.Append((byte)'\n').ToArray());
                    string? reason = GoldenSources.Check(header, temp, kind, py);
                    File.WriteAllBytes(path, original);
                    Assert.True(reason is not null && reason.Contains(rel), $"{kind}: a change of {rel} was not detected (reason: {reason})");
                }

                // A header with a missing or an extra file, or without the new fields, is stale.
                var withoutOne = HeaderFor(temp, kind, py);
                ((JsonObject)withoutOne["source_sha256"]!).Remove("csv_compat.py");
                Assert.Contains("file set", GoldenSources.Check(ToElement(withoutOne), temp, kind, py)!);
                var withExtra = HeaderFor(temp, kind, py);
                withExtra["source_sha256"]!["README.md"] = "00";
                Assert.Contains("file set", GoldenSources.Check(ToElement(withExtra), temp, kind, py)!);
                Assert.Contains("python_version", GoldenSources.Check(ToElement(new JsonObject { ["generator"] = "old" }), temp, kind, py)!);
                var noSources = new JsonObject { ["python_version"] = py };
                Assert.Contains("source_sha256", GoldenSources.Check(ToElement(noSources), temp, kind, py)!);
            }
            finally
            {
                Directory.Delete(temp, recursive: true);
            }
        }
    }

    [Fact]
    public void Generated_golden_files_carry_a_binding_that_matches_the_working_tree()
    {
        // The generators and GoldenSources must list the same files and compute the same hashes.
        Assert.Null(GoldenFiles.StaleReason(GoldenFiles.EnsureCardDb(), "carddb"));
        Assert.Null(GoldenFiles.StaleReason(GoldenFiles.EnsureRulesTrace(), "rules"));
        Assert.Null(GoldenFiles.StaleReason(GoldenFiles.EnsureSearchGolden("search"), "search"));
    }

    [Fact]
    public void A_golden_file_from_another_source_state_is_reported_stale()
    {
        // Same data, tampered binding: what a golden file generated from different Python source looks like.
        string copy = Path.Combine(Path.GetTempPath(), "ttc_stale_carddb_" + Guid.NewGuid().ToString("N") + ".json");
        try
        {
            var doc = JsonNode.Parse(File.ReadAllText(GoldenFiles.EnsureCardDb()))!.AsObject();
            Assert.Null(GoldenFiles.StaleReason(GoldenFiles.EnsureCardDb(), "carddb"));
            var sources = doc["header"]!["source_sha256"]!.AsObject();
            string first = sources.First().Key;
            sources[first] = new string('0', 64);
            File.WriteAllText(copy, doc.ToJsonString());
            string? reason = GoldenFiles.StaleReason(copy, "carddb");
            Assert.True(reason is not null && reason.Contains(first), $"stale golden not detected: {reason}");

            doc.AsObject()["header"]!.AsObject().Remove("source_sha256");
            File.WriteAllText(copy, doc.ToJsonString());
            Assert.Contains("source_sha256", GoldenFiles.StaleReason(copy, "carddb")!);

            doc.AsObject().Remove("header");
            File.WriteAllText(copy, doc.ToJsonString());
            Assert.Contains("no header", GoldenFiles.StaleReason(copy, "carddb")!);
        }
        finally
        {
            if (File.Exists(copy)) File.Delete(copy);
        }
    }
}
