import React, { useEffect, useRef, useState, useCallback } from "react";
import {
  Alert,
  Box,
  Button,
  Chip,
  CircularProgress,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Grid,
  LinearProgress,
  Stack,
  Typography,
} from "@mui/material";
import { jarvisApiClient } from "../../../../api/api";

/**
 * Props:
 *  - active: boolean            // start/stop polling
 *  - pollMs?: number            // default 2000
 *  - maxPreviews?: number       // default 3
 *  - onUpdate?: (statusObj) => void  // called on every poll with {status,current,total,dataset}
 *  - onTerminal?: (statusObj) => void // called once on completed/failed/cancelled
 */
const AnnotationJobMonitor = ({
  active,
  pollMs = 2000,
  maxPreviews = 3,
  onUpdate,
  onTerminal,
}) => {
  const pollRef = useRef(null);
  const previewsTickRef = useRef(0); // throttle previews (fetch every other tick)
  const [status, setStatus] = useState(null); // {status,current,total,dataset}
  const [previews, setPreviews] = useState([]);
  const [isLoadingPreviews, setIsLoadingPreviews] = useState(false);
  const [lightbox, setLightbox] = useState({ open: false, src: "", name: "" });
  const [error, setError] = useState(null);
  const [onceTerminal, setOnceTerminal] = useState(false);

  const datasetId = status?.dataset || status?.dataset_id || null;
  const isRunning = status?.status === "running";
  const isTerminal =
    status?.status === "completed" ||
    status?.status === "failed" ||
    status?.status === "cancelled";

  const percent =
    status?.total > 0
      ? Math.min(100, ((status.current || 0) / (status.total || 1)) * 100)
      : null;

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  useEffect(() => {
    // stop on unmount
    return stopPolling;
  }, [stopPolling]);

  const fetchPreviews = useCallback(
    async (ds) => {
      if (!ds) return;
      try {
        setIsLoadingPreviews(true);
        // 1) list images
        const { data } = await jarvisApiClient.get(
          `/api/db/datasets/${ds}/images`
        );
        const imgs = Array.isArray(data?.images)
          ? data.images.slice(-maxPreviews)
          : [];
        // 2) fetch annotated previews
        const fetched = [];
        for (const imgPath of imgs) {
          const { data: imgData } = await jarvisApiClient.get(
            `/api/db/datasets/${ds}/image`,
            { params: { path: imgPath } }
          );
          const b64 = imgData?.annotated?.base64;
          if (b64) {
            fetched.push({
              path: imgPath,
              src: `data:image/png;base64,${b64}`,
            });
          }
        }
        setPreviews(fetched);
      } catch (_) {
        // previews are non-fatal
      } finally {
        setIsLoadingPreviews(false);
      }
    },
    [maxPreviews]
  );

  const tick = useCallback(async () => {
    try {
      const { data } = await jarvisApiClient.get("/api/annotation/status");
      if (!data?.ok) return;

      const next = {
        status: data.annotation_status,
        current: data.current,
        total: data.total,
        dataset: data.dataset_id || data.dataset || null,
      };

      setStatus(next);
      setError(null);
      onUpdate?.(next);

      // throttle previews: fetch every other tick while running
      if (next.status === "running" && next.dataset) {
        previewsTickRef.current = (previewsTickRef.current + 1) % 2;
        if (previewsTickRef.current === 0) {
          fetchPreviews(next.dataset);
        }
      }

      if (
        (next.status === "completed" ||
          next.status === "failed" ||
          next.status === "cancelled") &&
        !onceTerminal
      ) {
        setOnceTerminal(true);
        onTerminal?.(next);
        stopPolling();
      }
    } catch (e) {
      setError(e?.response?.data?.error || e?.message || "Polling failed");
    }
  }, [fetchPreviews, onTerminal, onUpdate, onceTerminal, stopPolling]);

  // (re)start/stop polling based on "active"
  useEffect(() => {
    stopPolling();
    setError(null);
    setStatus(null);
    setPreviews([]);
    setOnceTerminal(false);
    if (active) {
      // fire immediately, then interval
      tick();
      pollRef.current = setInterval(tick, pollMs);
    }
  }, [active, pollMs, stopPolling, tick]);

  // UI
  return (
    <Stack gap={1} sx={{ mt: 2 }}>
      {/* Top line during upload/starting */}
      {!status && active && (
        <Stack direction="row" gap={1} alignItems="center">
          <CircularProgress size={18} />
          <Typography variant="body2">
            Uploading and starting annotation…
          </Typography>
        </Stack>
      )}

      {/* Error (non-fatal) */}
      {error && <Alert severity="error">{error}</Alert>}

      {/* Status row + progress */}
      {status && (
        <Stack gap={1}>
          <Stack direction="row" alignItems="center" gap={1}>
            <Typography variant="caption">Status:</Typography>
            <Chip
              size="small"
              label={status.status}
              color={
                status.status === "completed"
                  ? "success"
                  : status.status === "failed" || status.status === "cancelled"
                  ? "error"
                  : "default"
              }
            />
            {status.total > 0 && (
              <Typography variant="caption">
                ({status.current}/{status.total})
              </Typography>
            )}
          </Stack>
          <LinearProgress
            variant={percent != null ? "determinate" : "indeterminate"}
            value={percent != null ? percent : undefined}
          />
        </Stack>
      )}

      {/* Previews */}
      {datasetId && (
        <Box sx={{ mt: 1 }}>
          <Stack
            direction="row"
            alignItems="center"
            justifyContent="space-between"
            mb={1}
          >
            <Typography variant="subtitle2">Annotation Previews</Typography>
            {!!previews.length && (
              <Typography variant="caption" color="text.secondary">
                {previews.length} preview{previews.length > 1 ? "s" : ""}
              </Typography>
            )}
          </Stack>

          {previews.length > 0 ? (
            <>
              {isRunning && <LinearProgress sx={{ my: 0.5 }} />}
              <Grid container spacing={1.5}>
                {previews.map((img) => {
                  const name = img.path.split("/").pop() || img.path;
                  return (
                    <Grid key={img.path} item xs={12} sm={6} md={4}>
                      <Box
                        onClick={() =>
                          setLightbox({ open: true, src: img.src, name })
                        }
                        sx={{
                          cursor: "zoom-in",
                          borderRadius: 2,
                          overflow: "hidden",
                          bgcolor: "grey.100",
                          boxShadow: 1,
                          transition: "transform .2s ease, box-shadow .2s ease",
                          "&:hover": {
                            boxShadow: 3,
                            transform: "translateY(-2px)",
                          },
                        }}
                      >
                        <Box
                          component="img"
                          src={img.src}
                          alt={`Preview ${name}`}
                          sx={{
                            width: "100%",
                            height: 180,
                            objectFit: "contain",
                            display: "block",
                            backgroundColor: "grey.100",
                          }}
                        />
                        <Box
                          sx={{
                            px: 1,
                            py: 0.5,
                            borderTop: 1,
                            borderColor: "divider",
                            bgcolor: "background.paper",
                          }}
                        >
                          <Typography
                            variant="caption"
                            title={name}
                            sx={{
                              display: "block",
                              whiteSpace: "nowrap",
                              overflow: "hidden",
                              textOverflow: "ellipsis",
                            }}
                          >
                            {name}
                          </Typography>
                        </Box>
                      </Box>
                    </Grid>
                  );
                })}
              </Grid>
            </>
          ) : (
            <Alert severity="info" variant="outlined" sx={{ mt: 1 }}>
              {isLoadingPreviews
                ? "Loading previews…"
                : "No previews available yet — they’ll appear here as soon as the first images are annotated."}
            </Alert>
          )}

          {/* Lightbox */}
          <Dialog
            open={lightbox.open}
            onClose={() => setLightbox({ open: false, src: "", name: "" })}
            maxWidth="lg"
            fullWidth
          >
            <DialogTitle sx={{ pb: 0 }}>
              {lightbox.name || "Preview"}
            </DialogTitle>
            <DialogContent dividers>
              <Box
                component="img"
                src={lightbox.src}
                alt={lightbox.name || "Preview"}
                sx={{
                  width: "100%",
                  height: "80dvh",
                  display: "block",
                  borderRadius: 1,
                  backgroundColor: "grey.100",
                }}
              />
            </DialogContent>
            <DialogActions>
              <Button
                onClick={() => setLightbox({ open: false, src: "", name: "" })}
              >
                Close
              </Button>
            </DialogActions>
          </Dialog>
        </Box>
      )}
    </Stack>
  );
};

export default AnnotationJobMonitor;
