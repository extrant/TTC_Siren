using System.Net.Http;
using System.Text;
using System.Text.Json;
using TTC.Siren.Compat;
using TTC.Siren.Core;
using TTC.Siren.Explain;
using TTC.Siren.Server;

namespace TTC.Siren.Tests;

/// <summary>One move of the true game, as the simulator knows it (the log has to reconstruct this from the requests alone).</summary>
internal sealed record TrueMove(int Mover, int CardId, int Cell, List<(int Cell, int CardId, string Cause)> Flips);

/// <summary>
/// A complete game of Triple Triad played by the engine on both sides, for the game-log tests. The simulator holds the TRUE state (both hands). The
/// plugin user is one colour: for each of its turns the simulator builds the request exactly as the plugin would (the user's cards, the opponent's cards
/// as zeros unless the rules show them, the board with the swapped left/right of the protocol) and sends it to a server; the opponent's moves are chosen
/// by a separate unlogged service from the opponent's own point of view. The true timeline is recorded with the cause of every flip.
/// </summary>
internal sealed class GameSimulator
{
    private readonly CardDb _db;
    private readonly GameState _g;
    private readonly string _rulesText;
    private readonly int _user; // 0 red, 1 blue
    private readonly AiMoveService _opponentAi;
    public List<TrueMove> Truth { get; } = new();
    public List<(int Turn, string Json)> UserRequests { get; } = new();

    public GameSimulator(CardDb db, AiMoveService opponentAi, Random rng, string rulesText, int user, int first)
    {
        _db = db;
        _opponentAi = opponentAi;
        _rulesText = rulesText;
        _user = user;
        var rules = RuleParser.ParsePythonRuleString(rulesText).Rules;

        var pool = db.Cards.Where(c => c.Up + c.Right + c.Down + c.Left > 0 && db.FindFirstIdByEdges(c.Up, c.Right, c.Down, c.Left) == c.Id).ToList();
        var chosen = pool.OrderBy(_ => rng.Next()).Take(10).ToList();
        var snapshot = new StateSnapshot { Current = first, Rules = rules };
        for (int p = 0; p < 2; p++)
            foreach (var def in chosen.Skip(p * 5).Take(5))
                snapshot.Hands[p].Add(new CardSnapshot(def.Up, def.Right, def.Down, def.Left, (byte)p, def.Id, CardTypes.NameOf(def.TypeId)));
        _g = GameState.FromSnapshot(snapshot);
    }

    public GameState State => _g;
    public int User => _user;
    public bool Finished => _g.AvailableCellCount == 0;
    public bool UserToMove => _g.CurrentPlayer == _user;

    private JsonObjectBuilder RequestFor(int viewer, int maxDepth)
    {
        var board = new List<string>();
        for (int cell = 0; cell < 9; cell++)
        {
            int inst = _g.BoardInstance(cell);
            if (inst < 0) continue;
            // protocol: numU = up, numL = the database's RIGHT edge, numD = down, numR = the database's LEFT edge (D-011); owner 1 = blue, 2 = red
            board.Add($"{{\"pos\":[{cell / 3},{cell % 3}],\"numU\":{_g.InstanceBase(inst, 0)},\"numL\":{_g.InstanceBase(inst, 1)},\"numD\":{_g.InstanceBase(inst, 2)},\"numR\":{_g.InstanceBase(inst, 3)},\"owner\":{(_g.InstanceOwner(inst) == OwnerId.Red ? 2 : 1)}}}");
        }
        string Hand(int p, bool visible)
        {
            var items = new List<string>();
            for (int i = 0; i < _g.HandCount(p); i++)
            {
                int inst = _g.HandInstance(p, i);
                items.Add(visible
                    ? $"{{\"numU\":{_g.InstanceBase(inst, 0)},\"numL\":{_g.InstanceBase(inst, 1)},\"numD\":{_g.InstanceBase(inst, 2)},\"numR\":{_g.InstanceBase(inst, 3)},\"canUse\":true,\"inHand\":true}}"
                    : "{\"numU\":0,\"numL\":0,\"numD\":0,\"numR\":0,\"canUse\":true,\"inHand\":true}");
            }
            return "[" + string.Join(",", items) + "]";
        }
        bool open = _rulesText.Contains("全明牌");
        return new JsonObjectBuilder
        {
            Text = $"{{\"board\":[{string.Join(",", board)}],\"myHand\":{Hand(viewer, true)},\"oppHand\":{Hand(1 - viewer, open)},\"myOwner\":{(viewer == 0 ? 2 : 1)}," +
                   $"\"currentPlayer\":{(_g.CurrentPlayer == 0 ? 2 : 1)},\"rules\":\"{_rulesText}\",\"max_depth\":{maxDepth},\"max_time\":1000000",
        };
    }

    /// <summary>The body of the plugin's request for the user's turn (without the closing brace, so the caller can add fields).</summary>
    public string UserRequest(int maxDepth, string? extra = null) => RequestFor(_user, maxDepth).Text + (extra ?? "") + "}";

    /// <summary>Plays the user's move named by the server's answer (<c>card_id</c> and <c>pos</c>).</summary>
    public void PlayUser(JsonElement answer)
    {
        int cardId = answer.GetProperty("card_id").GetInt32();
        int row = answer.GetProperty("pos")[0].GetInt32(), col = answer.GetProperty("pos")[1].GetInt32();
        Play(cardId, row * 3 + col);
    }

    /// <summary>The opponent moves: chosen by an unlogged engine from the opponent's own point of view.</summary>
    public async Task PlayOpponentAsync()
    {
        int opp = 1 - _user;
        var request = RequestFor(opp, 4).Text + "}";
        using var doc = JsonDocument.Parse(request);
        var outcome = await _opponentAi.HandleAsync(doc.RootElement, 1234, CancellationToken.None);
        if (outcome.Status != 200 || !outcome.Body.ContainsKey("pos")) throw new InvalidOperationException("the opponent engine found no move: " + PyJson.Serialize(outcome.Body));
        var pos = (List<object?>)outcome.Body["pos"]!;
        Play((int)outcome.Body["card_id"]!, (int)pos[0]! * 3 + (int)pos[1]!);
    }

    private void Play(int cardId, int cell)
    {
        int mover = _g.CurrentPlayer;
        int inst = -1;
        for (int i = 0; i < _g.HandCount(mover); i++)
            if (_g.InstanceId(_g.HandInstance(mover, i)) == cardId) { inst = _g.HandInstance(mover, i); break; }
        if (inst < 0) throw new InvalidOperationException($"card {cardId} is not in the hand of player {mover}");
        var flips = FlipAnalyzer.Analyze(FlipAnalyzer.ViewOf(_g), cell, FlipAnalyzer.ViewOfInstance(_g, inst), (byte)mover, _g.Rules)
            .Select(f => (f.Cell, f.CardId, Names.Wire(f.Cause))).ToList();
        _g.MakeMove(new Move(inst, cell));
        Truth.Add(new TrueMove(mover, cardId, cell, flips));
    }

    public (int Mine, int Theirs) FinalCount()
    {
        _g.CountCards(out int red, out int blue);
        return _user == 0 ? (red, blue) : (blue, red);
    }
}

/// <summary>A request body under construction (the closing brace is added by the caller).</summary>
internal sealed class JsonObjectBuilder
{
    public string Text { get; init; } = "";
}

internal static class SimHttp
{
    public static async Task<(int Status, string Text)> PostAsync(HttpClient client, string url, string body, string? key = null, ulong? seed = null)
    {
        var message = new HttpRequestMessage(HttpMethod.Post, url + "/ai_move") { Content = new StringContent(body, Encoding.UTF8, "application/json") };
        if (key is not null) message.Headers.Add("X-Test-Key", key);
        if (seed is not null) message.Headers.Add("X-Ttc-Test-Seed", seed.Value.ToString()); // needs ServerOptions.TestSeedHeader
        using var response = await client.SendAsync(message);
        return ((int)response.StatusCode, await response.Content.ReadAsStringAsync());
    }
}
