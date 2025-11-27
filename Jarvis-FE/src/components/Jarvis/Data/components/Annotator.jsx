import React, {
  useEffect,
  useMemo,
  useRef,
  useState,
  useCallback,
} from "react";
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Typography,
  Alert,
  Box,
  Button,
  Stack,
  CircularProgress,
  LinearProgress,
  Divider,
} from "@mui/material";
import ReactImageAnnotate from "@starwit/react-image-annotate";
import { useToast } from "../../../Toast/Toast";
import { jarvisApiClient } from "../../../../api/api";
import AnnotationJobMonitor from "../../Data/components/AnnotationLoader"; // <-- adjust relative path if needed

const ENABLED_TOOLS = ["select", "create-box"];
const MIN_GOLDEN = 5;
const POLL_MS = 2000;

// Small helper to ensure arrays are always arrays
const asArray = (v) => (Array.isArray(v) ? v : []);

// deterministic mapping: train/foo.jpg → images/foo.jpg
const annotatedToRawPath = (annotatedPath) => {
  if (!annotatedPath) return annotatedPath;
  return annotatedPath.replace(/^train\//i, "images/");
};

const AnnotatorDialog = ({
  open,
  onClose,
  safeDatasetId,
  classes,
  filteredImages, // iterate through these
}) => {
  const toast = useToast();
  const annotatorWrapRef = useRef(null);
  const lastExitPayloadRef = useRef(null);

  // --- SAFE inputs (never undefined) ---
  const safeClasses = asArray(classes);
  const safeList = asArray(filteredImages);

  // form
  const [newName, setNewName] = useState("");

  // progress across images
  const [currentIndex, setCurrentIndex] = useState(0);

  // load state
  const [isLoading, setIsLoading] = useState(false);
  const [loadError, setLoadError] = useState("");

  // current image data
  const [currentB64, setCurrentB64] = useState("");
  const [initialRegions, setInitialRegions] = useState([]); // regions[] for current image
  const [editedRegionsMap, setEditedRegionsMap] = useState({}); // { [imgPath]: regions[] }

  // golden selections: { [imgPath]: anns[] }
  const [acceptedAnns, setAcceptedAnns] = useState({});

  // job state (for re-annotation after submit)
  const [isReannotating, setIsReannotating] = useState(false);
  const [jobStatus, setJobStatus] = useState(null);

  const goldenCount = useMemo(
    () => Object.keys(acceptedAnns).length,
    [acceptedAnns]
  );

  // debug logger that won't crash
  const log = (...args) => {
    console.debug("[AnnotatorDialog]", ...args);
  };

  // tiny wait helper
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  // click the library Save button so onExit fires
  const clickInternalSaveAndWait = async () => {
    const root = annotatorWrapRef.current || document;
    try {
      const saveSvg = root.querySelector('svg[data-testid="SaveIcon"]');
      const saveBtn = saveSvg && saveSvg.closest("button");
      if (saveBtn) saveBtn.click();
    } catch {
      // ignore
    }
    await sleep(0);
  };

  // reset each time dialog opens or the list changes
  useEffect(() => {
    if (!open) return;
    log("OPEN dialog; resetting state. List length:", safeList.length);
    setNewName("");
    setCurrentIndex(0);
    setIsLoading(false);
    setLoadError("");
    setCurrentB64("");
    setInitialRegions([]);
    setEditedRegionsMap({});
    setAcceptedAnns({});
    setIsReannotating(false);
    setJobStatus(null);
  }, [open, safeList.length]); // only depend on length to avoid ref churn

  // fetch current image (and convert to regions)
  useEffect(() => {
    if (!open) return;
    if (safeList.length === 0) {
      log("No images to review.");
      return;
    }
    const imgPath = safeList[currentIndex];
    if (!imgPath) {
      log("No imgPath at index", currentIndex);
      return;
    }

    const run = async () => {
      log("Fetching image", { index: currentIndex, imgPath });
      setIsLoading(true);
      setLoadError("");
      setCurrentB64("");
      setInitialRegions([]);

      try {
        const { data } = await jarvisApiClient.get(
          `/api/db/datasets/${encodeURIComponent(safeDatasetId)}/image`,
          {
            params: { path: imgPath, source: "annotated" },
          }
        );

        log("API /image response", data);

        if (!data?.ok) throw new Error(data?.error || "Failed to load image");

        const b64 = data?.original?.base64 || data?.annotated?.base64 || "";
        const anns = asArray(data?.annotations);

        // Convert anns -> react-image-annotate regions
        const regions = anns.map((ann, idx) => {
          const [cid, xc, yc, w, h] = ann;
          const label = safeClasses?.[cid] ?? "Unknown";
          const _xc = Number(xc) || 0;
          const _yc = Number(yc) || 0;
          const _w = Number(w) || 0;
          const _h = Number(h) || 0;

          return {
            type: "box",
            x: _xc - _w / 2,
            y: _yc - _h / 2,
            w: _w,
            h: _h,
            cls: label,
            highlighted: false,
            editingLabels: false,
            color: "#f44336",
            id: `${imgPath}__ann_${idx}`, // unique & stable
          };
        });

        log("Prepared regions", regions);

        setCurrentB64(b64);
        setInitialRegions(regions);
      } catch (e) {
        const msg =
          e?.response?.data?.error || e?.message || "Failed to load image";
        log("LOAD ERROR", msg, e);
        setLoadError(msg);
      } finally {
        setIsLoading(false);
      }
    };

    run();
  }, [open, currentIndex, safeList, safeClasses, safeDatasetId]);

  // Build annotator payload from current state
  const annotatorImages = useMemo(() => {
    const imgPath = safeList[currentIndex];
    const safeRegions =
      asArray(editedRegionsMap[imgPath])?.length > 0
        ? editedRegionsMap[imgPath]
        : asArray(initialRegions);

    const result =
      imgPath && currentB64
        ? [
            {
              src: `data:image/png;base64,${currentB64}`,
              name: imgPath,
              regions: safeRegions, // must be array
            },
          ]
        : [];

    log("annotatorImages memo", {
      imgPath,
      hasB64: !!currentB64,
      regions: safeRegions,
    });
    return result;
  }, [safeList, currentIndex, currentB64, initialRegions, editedRegionsMap]);

  const handleAnnotatorExit = (output) => {
    try {
      const imgPath = safeList[currentIndex];
      const regions = asArray(output?.images?.[0]?.regions);
      lastExitPayloadRef.current = output;
      log("onExit captured regions", { imgPath, regions });
      if (imgPath) {
        setEditedRegionsMap((prev) => ({ ...prev, [imgPath]: regions }));
      }
    } catch (e) {
      log("onExit error", e);
    }
  };

  const goNext = async () => {
    await clickInternalSaveAndWait();
    setEditedRegionsMap((prev) => ({ ...prev })); // flush
    if (currentIndex < safeList.length - 1) {
      setCurrentIndex((i) => i + 1);
    } else {
      toast({ title: "Reached end of list", status: "info" });
    }
  };

  const goPrev = async () => {
    await clickInternalSaveAndWait();
    if (currentIndex > 0) setCurrentIndex((i) => i - 1);
  };

  // Convert regions -> anns for API
  const regionsToAnns = (regions) =>
    asArray(regions).map((r) => {
      const cid = Math.max(0, safeClasses.indexOf(r.cls));
      const xc = (Number(r.x) || 0) + (Number(r.w) || 0) / 2;
      const yc = (Number(r.y) || 0) + (Number(r.h) || 0) / 2;
      return [cid, xc, yc, Number(r.w) || 0, Number(r.h) || 0, 1.0];
    });

  const handleAccept = async () => {
    const imgPath = safeList[currentIndex];
    if (!imgPath) return;

    await clickInternalSaveAndWait();

    const exitRegions = asArray(
      lastExitPayloadRef.current?.images?.[0]?.regions
    );
    const edited = editedRegionsMap[imgPath];

    const regions =
      exitRegions.length > 0
        ? exitRegions
        : edited !== undefined
        ? asArray(edited)
        : asArray(initialRegions);

    const anns = regionsToAnns(regions);

    log("ACCEPT image", { imgPath, anns });

    const rawPath = annotatedToRawPath(imgPath);
    setAcceptedAnns((prev) => ({ ...prev, [rawPath]: anns }));
    goNext();
  };

  const handleReject = () => {
    log("REJECT image", safeList[currentIndex]);
    goNext();
  };

  // --- Re-annotate submit + monitor integration ---
  const handleSubmit = async () => {
    try {
      log("SUBMIT golden selections", acceptedAnns);

      const payload = {
        new_name: newName.trim(),
        golden_selections: acceptedAnns,
      };

      const { data } = await jarvisApiClient.post(
        `/api/datasets/${encodeURIComponent(safeDatasetId)}/reannotate`,
        payload
      );

      if (!data?.ok) {
        throw new Error(data?.error || "Failed to start re-annotation");
      }

      toast({ title: "Re-annotation started", status: "info" });

      // arm the monitor AFTER server acknowledged the queueing
      setTimeout(() => setIsReannotating(true), 200);
    } catch (e) {
      const msg =
        e?.response?.data?.error ||
        e?.message ||
        "Failed to start re-annotation";
      log("SUBMIT error", msg, e);
      toast({ title: msg, status: "error" });
    }
  };

  const canSubmit = newName.trim().length > 0 && goldenCount >= MIN_GOLDEN;

  // Monitor handlers (stable)
  const handleMonitorUpdate = useCallback((s) => {
    setJobStatus(s);
  }, []);

  const handleMonitorTerminal = useCallback(
    (s) => {
      setIsReannotating(false);
      if (s.status === "completed") {
        toast({ title: "Re-annotation completed 🎉", status: "success" });
        onClose(); // close the dialog when done
      } else if (s.status === "failed") {
        toast({ title: "Re-annotation failed", status: "error" });
      } else if (s.status === "cancelled") {
        toast({ title: "Re-annotation cancelled", status: "info" });
      }
    },
    [onClose, toast]
  );

  const cancelReannotation = useCallback(async () => {
    try {
      const { data } = await jarvisApiClient.post("/api/annotation/cancel");
      if (!data?.ok)
        throw new Error(data?.error || "Failed to cancel re-annotation");
      toast({ title: "Re-annotation cancelled", status: "info" });
      setIsReannotating(false);
    } catch (e) {
      toast({
        title: "Failed to cancel re-annotation",
        status: "error",
        description: e?.message,
      });
    }
  }, [toast]);

  // Only render annotator when we have everything and arrays are defined
  const readyForAnnotator =
    open &&
    !isLoading &&
    !loadError &&
    !isReannotating && // hide annotator while job runs
    annotatorImages.length === 1 &&
    typeof annotatorImages[0].src === "string" &&
    Array.isArray(annotatorImages[0].regions) &&
    Array.isArray(safeClasses);

  return (
    <Dialog
      open={open}
      onClose={() => (!isReannotating ? onClose() : null)} // prevent closing during job unless you cancel
      fullScreen
    >
      <DialogTitle>Re-annotate Dataset</DialogTitle>

      <DialogContent sx={{ height: "100%" }}>
        <Stack
          spacing={1}
          sx={{
            height: "100%",
          }}
        >
          <TextField
            label="New Dataset Name"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            fullWidth
            size="small"
            disabled={isReannotating}
          />

          <Stack direction="row" alignItems="center" spacing={2}>
            <Typography variant="body2">
              Accepted (golden): <strong>{goldenCount}</strong> / {MIN_GOLDEN}{" "}
              required
            </Typography>
            <Box sx={{ flex: 1 }}>
              <LinearProgress
                variant="determinate"
                value={Math.min((goldenCount / MIN_GOLDEN) * 100, 100)}
              />
            </Box>
            <Typography variant="body2" color="text.secondary">
              Reviewing {currentIndex + 1} / {safeList.length}
            </Typography>
          </Stack>

          {/* Annotator area (hidden while background job runs) */}
          {!isReannotating && (
            <>
              {safeList.length === 0 && (
                <Alert severity="info">No images to review.</Alert>
              )}

              {isLoading && (
                <Stack
                  alignItems="center"
                  justifyContent="center"
                  sx={{ minHeight: 240 }}
                >
                  <CircularProgress />
                </Stack>
              )}

              {loadError && <Alert severity="error">{loadError}</Alert>}

              {readyForAnnotator ? (
                <Box
                  ref={annotatorWrapRef}
                  sx={{
                    height: 0,
                    flexGrow: 1,
                  }}
                >
                  <ReactImageAnnotate
                    key={safeList[currentIndex]}
                    taskDescription="Edit annotations"
                    images={annotatorImages}
                    selectedImage={annotatorImages[0].src}
                    regionClsList={safeClasses}
                    enabledTools={ENABLED_TOOLS}
                    onExit={handleAnnotatorExit}
                    hideNext
                    hidePrev
                    hideFullScreen
                    hideSettings
                    hideHeaderText
                    hideClone
                  />
                </Box>
              ) : (
                !isLoading &&
                !loadError &&
                safeList.length > 0 && (
                  <Alert severity="info">Preparing annotator…</Alert>
                )
              )}

              {/* External controls */}
              <Stack direction="row" justifyContent="space-between">
                <Button onClick={goPrev} disabled={currentIndex === 0}>
                  Previous
                </Button>
                <Stack direction="row" spacing={1}>
                  <Button
                    variant="outlined"
                    color="error"
                    onClick={handleReject}
                  >
                    Reject
                  </Button>
                  <Button
                    variant="outlined"
                    color="success"
                    onClick={handleAccept}
                  >
                    Accept
                  </Button>
                </Stack>
                <Button
                  onClick={goNext}
                  disabled={currentIndex >= safeList.length - 1}
                >
                  Next
                </Button>
              </Stack>
            </>
          )}

          {/* Monitor area (visible while job runs) */}
          {isReannotating && (
            <>
              <Divider sx={{ my: 1 }} />
              <Typography variant="subtitle2">
                Re-annotation progress
              </Typography>
              <AnnotationJobMonitor
                active={isReannotating}
                pollMs={POLL_MS}
                onUpdate={handleMonitorUpdate}
                onTerminal={handleMonitorTerminal}
              />
            </>
          )}
        </Stack>
      </DialogContent>

      <DialogActions>
        {!isReannotating ? (
          <>
            <Button onClick={onClose}>Cancel</Button>
            <Button
              onClick={handleSubmit}
              disabled={!canSubmit}
              variant="contained"
            >
              Submit
            </Button>
          </>
        ) : (
          <>
            <Button
              onClick={cancelReannotation}
              color="error"
              variant="outlined"
            >
              Cancel Re-annotation
            </Button>
            <Button onClick={() => onClose?.()} disabled>
              Close
            </Button>
          </>
        )}
      </DialogActions>
    </Dialog>
  );
};

export default AnnotatorDialog;
