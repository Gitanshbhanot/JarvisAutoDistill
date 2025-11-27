import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useParams, useNavigate } from "react-router-dom";
import {
  Container,
  Stack,
  Typography,
  Alert,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Tabs,
  Tab,
  Box,
  Divider,
  IconButton,
  CircularProgress,
  TextField,
  Autocomplete,
  Chip,
} from "@mui/material";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  Cancel01Icon,
  ArrowLeft01Icon,
  ArrowRight01Icon,
  Copy01Icon,
  Download01Icon,
} from "@hugeicons/core-free-icons";
import { jarvisApiClient } from "../../../api/api";

import TopBar from "./components/TopBar";
import ImageGrid from "./components/ImageGrid";
import { useToast } from "../../Toast/Toast";
import AnnotatorDialog from "./components/Annotator";

const PER_PAGE = 24;

const DataDetail = () => {
  const { datasetId } = useParams();
  const navigate = useNavigate();
  const safeDatasetId = datasetId ?? "";

  // data
  const [images, setImages] = useState([]);
  const [classes, setClasses] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  // view & filter
  const [viewSource, setViewSource] = useState("annotated"); // 'annotated' | 'raw'
  const [query, setQuery] = useState("");
  const [page, setPage] = useState(1);
  const [selectedClasses, setSelectedClasses] = useState([]);

  // preview
  const [preview, setPreview] = useState(null); // { path, original, annotated, info }
  const [previewTab, setPreviewTab] = useState("annotated");
  const [previewLoading, setPreviewLoading] = useState(false);
  const previewCache = useRef(new Map());

  // delete
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [deleteInput, setDeleteInput] = useState("");
  const [isDeleting, setIsDeleting] = useState(false);

  // download
  const [isDownloading, setIsDownloading] = useState(false);
  const [isVideoDownloading, setIsVideoDownloading] = useState(false);

  // Re-Annotate
  const [reannotateOpen, setReannotateOpen] = useState(false);

  const toast = useToast();

  // ---------- Load dataset list ----------
  const load = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);
      const { data } = await jarvisApiClient.get(
        `/api/db/datasets/${encodeURIComponent(safeDatasetId)}/images`,
        { params: { source: viewSource } }
      );
      if (!data?.ok) throw new Error(data?.error || "Failed to load dataset");
      setImages(Array.isArray(data.images) ? data.images : []);
      setClasses(Array.isArray(data.class_names) ? data.class_names : []);
      setPage(1);
    } catch (e) {
      setError(e?.response?.data?.error || e?.message || "Failed to load");
    } finally {
      setIsLoading(false);
    }
  }, [safeDatasetId, viewSource]);

  useEffect(() => {
    load();
  }, [load]);

  // ---------- filter + paginate ----------
  const filteredImages = useMemo(() => {
    const q = query.trim().toLowerCase();
    return q ? images.filter((p) => p.toLowerCase().includes(q)) : images;
  }, [images, query]);

  const pageCount = Math.max(1, Math.ceil(filteredImages.length / PER_PAGE));
  const paginated = useMemo(() => {
    const start = (page - 1) * PER_PAGE;
    return filteredImages.slice(start, start + PER_PAGE);
  }, [filteredImages, page]);

  useEffect(() => {
    if (page > pageCount) setPage(pageCount);
  }, [page, pageCount]);

  const onRefresh = () => load();

  // ---------- preview helpers ----------
  const copyToClipboard = async (text) => {
    try {
      await navigator.clipboard.writeText(text);
      toast({ title: "Path copied to clipboard", status: "success" });
    } catch {
      toast({ title: "Could not copy path", status: "error" });
    }
  };

  const downloadBase64 = (b64, filename) => {
    try {
      const a = document.createElement("a");
      a.href = `data:image/png;base64,${b64}`;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
    } catch {
      toast({ title: "Download failed", status: "error" });
    }
  };

  const openPreview = async (path) => {
    setPreview(null);
    setPreviewLoading(true);

    const classesKey = selectedClasses.slice().sort().join(",");
    const key = `${viewSource}::${path}::${classesKey}`;

    try {
      if (previewCache.current.has(key)) {
        const cached = previewCache.current.get(key);
        setPreview({ path, ...cached });
        setPreviewTab(cached?.annotated?.base64 ? "annotated" : "original");
      } else {
        const params = { path, source: viewSource };
        if (selectedClasses.length > 0) params.classes = selectedClasses; // array -> repeated params via axios

        const { data } = await jarvisApiClient.get(
          `/api/db/datasets/${encodeURIComponent(safeDatasetId)}/image`,
          { params }
        );
        if (!data?.ok) throw new Error(data?.error || "Failed to fetch image");
        previewCache.current.set(key, data);
        setPreview({ path, ...data });
        setPreviewTab(data?.annotated?.base64 ? "annotated" : "original");
      }
    } catch (e) {
      toast({
        title:
          e?.response?.data?.error || e?.message || "Failed to fetch image",
        status: "error",
      });
    } finally {
      setPreviewLoading(false);
    }
  };

  // keyboard nav in preview
  const currentIndex = useMemo(() => {
    if (!preview) return -1;
    return filteredImages.findIndex((p) => p === preview.path);
  }, [filteredImages, preview]);

  useEffect(() => {
    if (!preview) return;
    const onKey = (e) => {
      if (e.key === "Escape") setPreview(null);
      if (e.key === "ArrowLeft" && currentIndex > 0)
        openPreview(filteredImages[currentIndex - 1]);
      if (e.key === "ArrowRight" && currentIndex < filteredImages.length - 1)
        openPreview(filteredImages[currentIndex + 1]);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [preview, currentIndex, filteredImages]);

  // ---------- delete ----------
  const deleteDataset = async () => {
    try {
      setIsDeleting(true);
      const { data } = await jarvisApiClient.delete(
        `/api/datasets/${encodeURIComponent(safeDatasetId)}`
      );
      if (!data?.ok) throw new Error(data?.error || "Failed to delete dataset");
      toast({ title: "Dataset deleted", status: "success" });
      setDeleteOpen(false);
      navigate(-1);
    } catch (e) {
      toast({
        title: e?.response?.data?.error || e?.message || "Failed to delete",
        status: "error",
      });
    } finally {
      setIsDeleting(false);
    }
  };

  // ---------- downloads ----------
  const handleDownload = async () => {
    setIsDownloading(true);
    try {
      const response = await jarvisApiClient.get(
        `/api/datasets/${encodeURIComponent(safeDatasetId)}/download`,
        { responseType: "blob" }
      );
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", `${safeDatasetId}.zip`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      toast({ title: "Dataset downloaded", status: "success" });
    } catch (e) {
      toast({ title: e?.message || "Download failed", status: "error" });
    } finally {
      setIsDownloading(false);
    }
  };

  const handleVideoDownload = async () => {
    setIsVideoDownloading(true);
    try {
      const response = await jarvisApiClient.get(
        `/api/datasets/${encodeURIComponent(safeDatasetId)}/annotated_video`,
        { responseType: "blob" }
      );
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", `${safeDatasetId}_annotated.mp4`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      toast({ title: "Annotated video downloaded", status: "success" });
    } catch (e) {
      toast({ title: e?.message || "Video download failed", status: "error" });
    } finally {
      setIsVideoDownloading(false);
    }
  };

  // invalidate preview cache on filter/source change
  useEffect(() => {
    previewCache.current = new Map();
  }, [selectedClasses, viewSource]);

  // ---------- UI ----------
  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <TopBar
        datasetId={safeDatasetId}
        classes={classes}
        viewSource={viewSource}
        onViewSourceChange={setViewSource}
        query={query}
        onQueryChange={(v) => {
          setQuery(v);
          setPage(1);
        }}
        isLoading={isLoading}
        onRefresh={onRefresh}
        onBack={() => navigate(-1)}
        onDelete={() => {
          setDeleteInput("");
          setDeleteOpen(true);
        }}
        onDownload={handleDownload}
        isDownloading={isDownloading}
        onVideoDownload={handleVideoDownload}
        isVideoDownloading={isVideoDownloading}
        onReannotate={() => setReannotateOpen(true)}
      />
      <Stack direction="row" gap={2} sx={{ mb: 2 }} alignItems="center">
        <Autocomplete
          multiple
          options={classes}
          value={selectedClasses}
          onChange={(_, v) => setSelectedClasses(v)}
          renderTags={(value, getTagProps) =>
            value.map((option, index) => (
              <Chip
                variant="outlined"
                label={option}
                {...getTagProps({ index })}
              />
            ))
          }
          size="small"
          renderInput={(params) => (
            <TextField
              {...params}
              label="Show classes (annotated view)"
              placeholder="All"
            />
          )}
          sx={{ minWidth: 360 }}
          disabled={viewSource === "raw"}
        />
      </Stack>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <ImageGrid
        items={paginated}
        filteredCount={filteredImages.length}
        page={page}
        pageCount={pageCount}
        onPageChange={setPage}
        isLoading={isLoading}
        viewSource={viewSource}
        query={query}
        onOpenPreview={openPreview}
      />

      {/* Preview dialog */}
      <Dialog
        open={!!preview || previewLoading}
        onClose={() => setPreview(null)}
        maxWidth="lg"
        fullWidth
      >
        <DialogTitle sx={{ pr: 7 }}>
          {preview?.path || "Loading…"}
          <IconButton
            aria-label="close"
            onClick={() => setPreview(null)}
            sx={{ position: "absolute", right: 8, top: 8 }}
          >
            <HugeiconsIcon icon={Cancel01Icon} />
          </IconButton>
        </DialogTitle>
        <DialogContent dividers>
          {previewLoading ? (
            <Stack
              alignItems="center"
              justifyContent="center"
              sx={{ minHeight: "40vh" }}
            >
              <CircularProgress />
            </Stack>
          ) : preview ? (
            <Stack gap={2}>
              <Stack
                direction="row"
                alignItems="center"
                justifyContent="space-between"
                gap={1}
              >
                <Tabs
                  value={previewTab}
                  onChange={(_, v) => setPreviewTab(v)}
                  aria-label="preview"
                >
                  <Tab
                    value="annotated"
                    label="Annotated"
                    disabled={!preview?.annotated?.base64}
                  />
                  <Tab
                    value="original"
                    label="Original"
                    disabled={!preview?.original?.base64}
                  />
                </Tabs>
                <Stack direction="row" gap={1} alignItems="center">
                  <IconButton
                    onClick={() =>
                      preview?.path && copyToClipboard(preview.path)
                    }
                    title="Copy path"
                  >
                    <HugeiconsIcon icon={Copy01Icon} />
                  </IconButton>
                  {previewTab === "annotated" && preview?.annotated?.base64 && (
                    <IconButton
                      onClick={() =>
                        downloadBase64(
                          preview.annotated.base64,
                          "annotated.png"
                        )
                      }
                      title="Download annotated"
                    >
                      <HugeiconsIcon icon={Download01Icon} />
                    </IconButton>
                  )}
                  {previewTab === "original" && preview?.original?.base64 && (
                    <IconButton
                      onClick={() =>
                        downloadBase64(preview.original.base64, "original.png")
                      }
                      title="Download original"
                    >
                      <HugeiconsIcon icon={Download01Icon} />
                    </IconButton>
                  )}
                </Stack>
              </Stack>

              <Box
                sx={{
                  border: "1px solid",
                  borderColor: "divider",
                  borderRadius: 1,
                  p: 1,
                  bgcolor: "background.paper",
                }}
              >
                {previewTab === "annotated" && preview?.annotated?.base64 ? (
                  <img
                    src={`data:image/png;base64,${preview.annotated.base64}`}
                    alt="annotated"
                    style={{
                      maxHeight: "40dvh",
                      width: "100%",
                      objectFit: "contain",
                      display: "block",
                    }}
                  />
                ) : previewTab === "original" && preview?.original?.base64 ? (
                  <img
                    src={`data:image/png;base64,${preview.original.base64}`}
                    alt="original"
                    style={{
                      maxHeight: "40dvh",
                      width: "100%",
                      objectFit: "contain",
                      display: "block",
                    }}
                  />
                ) : (
                  <Stack
                    alignItems="center"
                    justifyContent="center"
                    sx={{ minHeight: 240 }}
                  >
                    <Typography variant="body2" color="text.secondary">
                      Not available
                    </Typography>
                  </Stack>
                )}
              </Box>

              {preview?.info && (
                <>
                  <Divider />
                  <Typography variant="body2" sx={{ whiteSpace: "pre-wrap" }}>
                    {preview.info}
                  </Typography>
                </>
              )}
            </Stack>
          ) : null}
        </DialogContent>
        <DialogActions sx={{ justifyContent: "space-between" }}>
          <Stack direction="row" gap={1}>
            <Button
              startIcon={<HugeiconsIcon icon={ArrowLeft01Icon} />}
              onClick={() =>
                currentIndex > 0 &&
                openPreview(filteredImages[currentIndex - 1])
              }
              disabled={currentIndex <= 0 || previewLoading}
            >
              Previous
            </Button>
            <Button
              endIcon={<HugeiconsIcon icon={ArrowRight01Icon} />}
              onClick={() =>
                currentIndex < filteredImages.length - 1 &&
                openPreview(filteredImages[currentIndex + 1])
              }
              disabled={
                currentIndex < 0 ||
                currentIndex >= filteredImages.length - 1 ||
                previewLoading
              }
            >
              Next
            </Button>
          </Stack>
          <Button
            startIcon={<HugeiconsIcon icon={Cancel01Icon} />}
            onClick={() => setPreview(null)}
          >
            Close
          </Button>
        </DialogActions>
      </Dialog>

      {/* Delete confirm */}
      <Dialog
        open={deleteOpen}
        onClose={() => (!isDeleting ? setDeleteOpen(false) : null)}
        maxWidth="xs"
        fullWidth
      >
        <DialogTitle>Delete dataset</DialogTitle>
        <DialogContent dividers>
          <Alert severity="warning" sx={{ mb: 2 }}>
            This action cannot be undone. Type the dataset id to confirm:
            <Box
              component="span"
              sx={{ display: "block", mt: 1, fontFamily: "monospace" }}
            >
              {safeDatasetId}
            </Box>
          </Alert>
          <TextField
            autoFocus
            label="Confirm dataset id"
            value={deleteInput}
            onChange={(e) => setDeleteInput(e.target.value)}
            disabled={isDeleting}
            fullWidth
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteOpen(false)} disabled={isDeleting}>
            Cancel
          </Button>
          <Button
            color="error"
            variant="contained"
            onClick={deleteDataset}
            disabled={isDeleting || deleteInput !== safeDatasetId}
          >
            {isDeleting ? "Deleting…" : "Delete"}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Re-annotate Dialog (ReactImageAnnotate) */}
      <AnnotatorDialog
        open={reannotateOpen}
        onClose={() => setReannotateOpen(false)}
        safeDatasetId={safeDatasetId}
        classes={classes}
        filteredImages={filteredImages}
      />
    </Container>
  );
};

export default DataDetail;
