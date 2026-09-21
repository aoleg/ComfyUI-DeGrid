using System;
using System.Collections.Generic;
using System.Linq;
using Newtonsoft.Json.Linq;
using SwarmUI.Builtin_ComfyUIBackend;
using SwarmUI.Core;
using SwarmUI.Text2Image;
using SwarmUI.Utils;

namespace Aoleg.DeGrid;

/// <summary>SwarmUI extension for ComfyUI-DeGrid: removes the 2px pixel grid the Qwen Image / Wan 2.1 VAEs
/// leave on decoded images (Krea 2, Qwen Image, Qwen Image 2.1, Anima). The node pack is this same repo,
/// installed on the backend through the feature button; this class only inserts the node into the workflow.</summary>
public class DeGridExtension : Extension
{
    public const string FeatureId = "degrid";

    public const string NodeId = "VAEDeGrid";

    /// <summary>Repo the backend node pack is installed from: this one. SwarmUI sends node inputs by name and
    /// ComfyUI silently drops any the installed node does not declare, so the node and this extension must be
    /// the same commit. SwarmUI clones this into DLNodes/ComfyUI-DeGrid, skips the clone if that folder already
    /// exists, and git-pulls it on every startup.</summary>
    public const string RepoUrl = "https://github.com/aoleg/ComfyUI-DeGrid";

    /// <summary>Node inputs this extension sends. Anything missing on the installed node means an older node pack
    /// (or a same-named node from another pack) and is reported by <see cref="WarnOnNodeVersionMismatch"/>.</summary>
    public static string[] RequiredNodeInputs = ["image", "enabled", "mode", "limit", "grid_gain", "grid_view", "skip_when_clean", "threshold"];

    /// <summary>Process-wide init claim. The feature install clones this repo into DLNodes/, and SwarmUI's compile
    /// glob does not exclude DLNodes, so this file is compiled twice: once into the core assembly from that clone and
    /// once into the extension assembly from src/Extensions/. Both copies get OnInit(). They are different types in
    /// different assemblies, so a static field would not be shared; AppDomain data is one slot for the whole process.</summary>
    public const string InitClaimKey = "Aoleg.DeGrid.DeGridExtension.Initialized";

    public static T2IRegisteredParam<string> Mode;

    public static T2IRegisteredParam<double> Limit, Threshold;

    public static T2IRegisteredParam<bool> SkipWhenClean;

    public static T2IParamGroup DeGridGroup;

    static bool ClaimInit()
    {
        if (AppDomain.CurrentDomain.GetData(InitClaimKey) is not null)
        {
            return false;
        }
        AppDomain.CurrentDomain.SetData(InitClaimKey, InitClaimKey);
        return true;
    }

    public override void PopulateMetadata()
    {
        base.PopulateMetadata();
        Description = "Removes the 2px pixel grid the Qwen Image / Wan 2.1 VAEs leave on decoded images (Krea 2, Qwen Image, Qwen Image 2.1, Anima).";
        Tags = ["parameters", "nodes"];
        License = "Apache-2.0";
    }

    public override void OnInit()
    {
        // Idempotent registrations stay outside the guard so the feature exists whichever copy runs first.
        InstallableFeatures.RegisterInstallableFeature(new("VAE DeGrid", FeatureId, RepoUrl, "aoleg", $"This will install the ComfyUI-DeGrid node pack from {RepoUrl} (Apache-2.0).\nIt must be the same repo this extension came from, so the node and the extension stay in step.\nDo you wish to install?"));
        ComfyUIBackendExtension.NodeToFeatureMap[NodeId] = FeatureId;
        if (!ClaimInit())
        {
            Logs.Debug("[DeGrid] DeGridExtension.OnInit() ran again (second copy compiled from the DLNodes clone); skipping duplicate registration.");
            return;
        }
        ScriptFiles.Add("assets/degrid_install.js");
        ComfyUIBackendExtension.RawObjectInfoParsers.Add(WarnOnNodeVersionMismatch);
        DeGridGroup = new("VAE DeGrid", Toggles: true, Open: false, IsAdvanced: true,
            Description: "Removes the 2px pixel grid the Qwen Image / Wan 2.1 VAEs leave on decoded images (Krea 2, Qwen Image, Qwen Image 2.1, Anima).\nRuns right after the VAE decode, before segmentation, video steps and SeedVR, and again after a pixel-space refiner decode so the upscaler never sees the grid.\nImages with no measurable grid are passed through untouched, so it is safe to leave on for other models.");
        Mode = T2IParamTypes.Register<string>(new("[DeGrid] Mode", "[DeGrid]\n'auto' measures the grid strength of each image and sets the removal limit itself - nothing to tune.\n'manual' uses '[DeGrid] Limit' instead; use it only if auto visibly under- or over-corrects.",
            "auto", Group: DeGridGroup, FeatureFlag: FeatureId, OrderPriority: 1,
            GetValues: (_) => ["auto///Auto (measure and calibrate per image)", "manual///Manual (use Limit)"]
            ));
        Limit = T2IParamTypes.Register<double>(new("[DeGrid] Limit", "[DeGrid]\nManual mode only. Maximum per-pixel correction on the 0-1 scale; the VAE grid is usually 0.005-0.02.\nToo low and the grid partially survives in contrasty areas (the backend log then says 'partially removed'). Too high and fine 2-3px texture (pores, fabric) gets slightly softened.",
            "0.02", Min: 0, Max: 0.1, Step: 0.001, Group: DeGridGroup, FeatureFlag: FeatureId, OrderPriority: 2,
            Examples: ["0.01", "0.02", "0.04"]
            ));
        SkipWhenClean = T2IParamTypes.Register<bool>(new("[DeGrid] Skip When Clean", "[DeGrid]\nLeave the image completely untouched when no phase-locked lattice is measured, eg anything that already went through an upscaler or a resize.\nTurn off only to force the filter to run regardless.",
            "true", Group: DeGridGroup, FeatureFlag: FeatureId, OrderPriority: 3
            ));
        Threshold = T2IParamTypes.Register<double>(new("[DeGrid] Clean Threshold", "[DeGrid]\nLattice amplitude, in /255 units, below which an image counts as clean.\nNative Qwen-VAE decodes measure about 0.5-2.5; upscaled or resized images about 0.1-0.2. Lower it (0.3) if a model you know is gridded reports 'none detected' in the backend log.",
            "0.5", Min: 0.05, Max: 5, Step: 0.05, Group: DeGridGroup, FeatureFlag: FeatureId, OrderPriority: 4,
            Examples: ["0.3", "0.5", "1"]
            ));
        // Refiner region ends at -4 and the core final decode ("8") runs at 1.
        WorkflowGenerator.AddStep(ApplyToRefinerDecode, -3.9);
        WorkflowGenerator.AddStep(ApplyToFinalDecode, 1.5);
    }

    /// <summary>Warns when the installed node does not accept everything this extension sends. ComfyUI builds a node's
    /// arguments from its own schema and ignores the rest of the prompt, so a stale node pack turns a parameter into a
    /// silent no-op. Also catches a same-named node from a different pack (ComfyUI-SaveSimple bundles one).</summary>
    public static void WarnOnNodeVersionMismatch(JObject rawObjectInfo)
    {
        if (rawObjectInfo[NodeId] is not JObject node)
        {
            return; // not installed; the feature flag covers that
        }
        HashSet<string> declared = [];
        foreach (string section in new[] { "required", "optional" })
        {
            if (node["input"]?[section] is JObject inputs)
            {
                foreach (JProperty prop in inputs.Properties())
                {
                    declared.Add(prop.Name);
                }
            }
        }
        string[] missing = [.. RequiredNodeInputs.Where(i => !declared.Contains(i))];
        if (missing.Length > 0)
        {
            Logs.Warning($"[DeGrid] The installed '{NodeId}' node does not accept {string.Join(", ", missing)}. "
                + "ComfyUI ignores inputs a node does not declare, so those options will do nothing (with no error). "
                + $"Update the node pack in DLNodes - its git remote must be {RepoUrl} - and make sure no other pack (eg ComfyUI-SaveSimple) supplies a node with the same id.");
        }
    }

    /// <summary>True when the group toggle is on. Throws if the backend lacks the node.</summary>
    static bool IsEnabled(WorkflowGenerator g)
    {
        // Any one group-param being present indicates the group toggle is enabled (all params in the group toggle as one).
        if (!g.UserInput.TryGet(Mode, out _))
        {
            return false;
        }
        if (!g.Features.Contains(FeatureId))
        {
            throw new SwarmUserErrorException("VAE DeGrid parameters specified, but the ComfyUI-DeGrid node pack isn't installed on the backend.");
        }
        return true;
    }

    static string MetaString(WorkflowGenerator g)
    {
        string mode = g.UserInput.Get(Mode, "auto");
        string limit = mode == "manual" ? $";limit={g.UserInput.Get(Limit, 0.02)}" : "";
        return $"mode={mode}{limit};skip={(g.UserInput.Get(SkipWhenClean, true) ? 1 : 0)};threshold={g.UserInput.Get(Threshold, 0.5)}";
    }

    static string CreateDeGridNode(WorkflowGenerator g, JArray image)
    {
        return g.CreateNode(NodeId, new JObject()
        {
            ["image"] = image,
            ["enabled"] = true,
            ["mode"] = g.UserInput.Get(Mode, "auto"),
            ["limit"] = g.UserInput.Get(Limit, 0.02),
            ["skip_when_clean"] = g.UserInput.Get(SkipWhenClean, true),
            ["threshold"] = g.UserInput.Get(Threshold, 0.5),
            ["grid_gain"] = 10.0,
            ["grid_view"] = "full frame"
        });
    }

    /// <summary>After the core final decode (node "8", priority 1) and its mask composite; before segmentation (5),
    /// video (10+) and SeedVR (40). Whatever decoded the image - the checkpoint VAE, a Pixel Decoder model, an
    /// upscaling decoder - the node measures the result and skips it when there is nothing to remove.</summary>
    public static void ApplyToFinalDecode(WorkflowGenerator g)
    {
        if (!IsEnabled(g))
        {
            return;
        }
        if (g.CurrentMedia.DataType != WGNodeData.DT_IMAGE)
        {
            Logs.Debug($"[DeGrid] final media is {g.CurrentMedia.DataType}, not an image; skipped.");
            return;
        }
        string node = CreateDeGridNode(g, g.CurrentMedia.Path);
        g.CurrentMedia = g.CurrentMedia.WithPath([node, 0]);
        g.UserInput.ExtraMeta["degrid"] = MetaString(g);
    }

    /// <summary>The refiner step (-4) decodes to the reserved node "24" when it upscales in pixel space, saves an
    /// intermediate, or composites a mask, then feeds those pixels to the upscaler ("26"/"28"), the save ("29") and
    /// the re-encode ("25") with no hook in between. Insert the node after "24" and re-point every consumer, so the
    /// upscaler is not handed a lattice it would read as detail. When the latent came straight from an encode of
    /// pixels the generator skips that decode and "24" does not exist, and there is nothing to do.</summary>
    public static void ApplyToRefinerDecode(WorkflowGenerator g)
    {
        if (!IsEnabled(g) || !g.HasNode("24"))
        {
            return;
        }
        string node = CreateDeGridNode(g, WorkflowGenerator.NodePath("24", 0));
        int rerouted = 0;
        foreach (JProperty nodeProp in g.Workflow.Properties().ToList())
        {
            if (nodeProp.Name == node || nodeProp.Value["inputs"] is not JObject inputs)
            {
                continue;
            }
            foreach (JProperty input in inputs.Properties().ToList())
            {
                if (input.Value is JArray link && link.Count == 2 && $"{link[0]}" == "24" && $"{link[1]}" == "0")
                {
                    input.Value = new JArray() { node, 0 };
                    rerouted++;
                }
            }
        }
        if (g.CurrentMedia.Path is JArray current && current.Count == 2 && $"{current[0]}" == "24" && $"{current[1]}" == "0")
        {
            g.CurrentMedia = g.CurrentMedia.WithPath([node, 0]);
            rerouted++;
        }
        if (rerouted == 0)
        {
            g.Workflow.Remove(node); // decode exists but nothing reads it as pixels here; don't leave a dangling node
            return;
        }
        Logs.Debug($"[DeGrid] refiner decode: {rerouted} consumer(s) of node 24 now read the degridded image.");
        g.UserInput.ExtraMeta["degrid"] = MetaString(g);
    }
}
