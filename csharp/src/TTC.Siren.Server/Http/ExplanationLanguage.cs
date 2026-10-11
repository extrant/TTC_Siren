using System.Text.Json.Nodes;
using TTC.Siren.Explain;

namespace TTC.Siren.Server;

/// <summary>
/// The language of the explanations (decision A-29). The facts of a decision are the same in every language; the game log keeps the text in every language
/// (<c>explanation.texts</c>) and the live <c>decision</c> events carry the brief in every language (<c>briefs</c>), so a replay or a stream can be read in either by
/// choosing at read time (<c>?lang=en</c>): the chosen text takes the place of <c>explanation.text</c> / <c>brief</c>. Nothing else is touched and no number is rewritten.
/// </summary>
public static class ExplanationLanguage
{
    /// <summary>The language a request asks for (<c>?lang=</c>), or null when it asks for none (or for one we have no templates for).</summary>
    public static string? Requested(Microsoft.AspNetCore.Http.HttpRequest request) => TemplateSet.Normalise(request.Query["lang"].ToString());

    /// <summary>One line of a game log with the explanation text in <paramref name="language"/>; the line itself when it has no explanation or no such text.</summary>
    public static string RewriteRecord(string line, string language)
    {
        if (!line.Contains("\"texts\"", StringComparison.Ordinal)) return line;
        var node = JsonNode.Parse(line);
        return Apply(node?["explanation"], language) ? node!.ToJsonString() : line;
    }

    /// <summary>The data of a live event with the text in <paramref name="language"/> (<c>game.updated</c> carries a record, <c>decision</c> the brief).</summary>
    public static string RewriteEvent(string json, string type, string language)
    {
        if (type == "game.updated" && json.Contains("\"texts\"", StringComparison.Ordinal))
        {
            var node = JsonNode.Parse(json);
            return Apply(node?["record"]?["explanation"], language) ? node!.ToJsonString() : json;
        }
        if (type == "decision" && json.Contains("\"briefs\"", StringComparison.Ordinal))
        {
            var node = JsonNode.Parse(json);
            if (node?["briefs"]?[language] is JsonNode brief)
            {
                node["brief"] = brief.DeepClone();
                node["language"] = language;
                return node.ToJsonString();
            }
        }
        return json;
    }

    private static bool Apply(JsonNode? explanation, string language)
    {
        if (explanation is not JsonObject obj || obj["texts"]?[language] is not JsonNode text) return false;
        obj["text"] = text.DeepClone();
        obj["language"] = language;
        return true;
    }
}
