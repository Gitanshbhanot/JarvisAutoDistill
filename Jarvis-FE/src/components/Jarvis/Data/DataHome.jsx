import React, {
  useEffect,
  useMemo,
  useRef,
  useState,
  useCallback,
} from "react";
import { useNavigate } from "react-router-dom";
import {
  Box,
  Button,
  Card,
  CardActionArea,
  CardContent,
  Chip,
  Container,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Grid,
  IconButton,
  LinearProgress,
  Stack,
  TextField,
  Tooltip,
  Typography,
  Alert,
  Divider,
  CircularProgress,
  Skeleton,
  Slider,
  Select,
  MenuItem,
  FormHelperText,
  FormControlLabel,
  Checkbox,
  InputAdornment,
} from "@mui/material";
import {
  Add01Icon,
  Cancel01Icon,
  RefreshIcon,
  CloudUploadIcon,
  Folder01Icon,
} from "@hugeicons/core-free-icons";
import { jarvisApiClient, jarvisApiClientFormData } from "../../../api/api";
import { HugeiconsIcon } from "@hugeicons/react";
import { useToast } from "../../Toast/Toast";
import { modelOptions } from "../Constants";
import AnnotationJobMonitor from "./components/AnnotationLoader";

const POLL_MS = 2000;

const DataHome = () => {
  const navigate = useNavigate();

  // Listing state
  const [dataSets, setDataSets] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  // Upload state
  const [showUploader, setShowUploader] = useState(false);
  const [objectName, setObjectName] = useState("");
  const [datasetName, setDatasetName] = useState("");
  const [zipFile, setZipFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState(null);
  const [annotationStatus, setAnnotationStatus] = useState(null);
  const [problemStatement, setProblemStatement] = useState("");
  const [sampleFiles, setSampleFiles] = useState([]);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.8);
  const [isEnhancing, setIsEnhancing] = useState(false);
  const [imageSize, setImageSize] = useState(512);
  const [upscaleImage, setUpscaleImage] = useState(false);

  // UI state
  const toast = useToast();

  // Model state (provider-qualified)
  const [modelName, setModelName] = useState("gemini:gemini-2.5-flash-lite");

  const pollRef = useRef(null);
  const fileInputRef = useRef(null);
  const sampleInputRef = useRef(null);

  const canSubmit = useMemo(() => {
    if (sampleFiles.length > 3) {
      setUploadError("Maximum 3 sample images allowed");
      return false;
    }

    // Check if reasoning model is selected and problem statement is required
    const isReasoningModel = modelName?.startsWith("reasoning:");
    if (isReasoningModel && !problemStatement.trim()) {
      setUploadError("Problem statement is required for reasoning models");
      return false;
    }

    return !!objectName.trim() && !!zipFile && !isUploading;
  }, [
    objectName,
    zipFile,
    isUploading,
    sampleFiles,
    modelName,
    problemStatement,
  ]);

  const fetchDatasets = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);
      const { data } = await jarvisApiClient.get("/api/datasets");
      if (!data.ok) throw new Error(data.error || "Failed to load datasets");
      setDataSets(Array.isArray(data.datasets) ? data.datasets : []);
    } catch (e) {
      const msg =
        e?.response?.data?.error || e?.message || "Failed to load datasets";
      setError(msg);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchDatasets();
  }, [fetchDatasets]);

  // Clear polling on unmount
  useEffect(() => {
    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, []);

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!canSubmit || !zipFile) return;

    if (zipFile.size > 5 * 1024 * 1024 * 1024) {
      setUploadError("ZIP file too large (max 5GB)");
      toast({ title: "File too large", status: "error" });
      return;
    }

    setUploadError(null);
    setAnnotationStatus({
      status: "pending",
      current: 0,
      total: 0,
      dataset: null,
    });

    try {
      // Step 1: upload zip
      const form = new FormData();
      form.append("object_name", objectName.trim());
      form.append("file", zipFile);
      if (datasetName.trim()) form.append("dataset_name", datasetName.trim());
      form.append("confidence_threshold", confidenceThreshold.toString());
      if (problemStatement.trim())
        form.append("problem_statement", problemStatement.trim());
      form.append("model_name", modelName); // pass provider-qualified model
      form.append("image_size", imageSize.toString());
      form.append("upscale_image", upscaleImage.toString());
      sampleFiles.forEach((file) => form.append("samples", file));

      const { data: upData } = await jarvisApiClientFormData.post(
        "/api/upload_zip",
        form
      );
      if (!upData?.ok) throw new Error(upData?.error || "Upload failed");

      // Step 2: start annotation
      const { data: annData } = await jarvisApiClient.post(
        "/api/start_annotation",
        {
          object_name: objectName.trim(),
          confidence_threshold: confidenceThreshold,
          model_name: modelName, // pass provider-qualified model
          image_size: imageSize,
          upscale_image: upscaleImage,
        }
      );
      if (!annData?.ok)
        throw new Error(annData?.error || "Could not start annotation");

      toast({ title: "Annotation started…", status: "info" });
      // Step 3: now activate the monitor (prevents seeing stale 'completed')
      setTimeout(() => setIsUploading(true), 250); // tiny buffer; optional
    } catch (e) {
      const msg =
        e?.response?.data?.error || e?.message || "Something went wrong";
      setUploadError(msg);
      setDatasetName(""); // Reset dataset name on error
    }
  };

  // Cancel current annotation job
  const handleCancelAnnotation = async () => {
    try {
      const { data } = await jarvisApiClient.post("/api/annotation/cancel");
      if (!data?.ok)
        throw new Error(data?.error || "Failed to cancel annotation");
      toast({
        title: "Annotation cancelled",
        status: "info",
        description: data.message,
      });
      setAnnotationStatus(null);
      setIsUploading(false);
      setZipFile(null);
      setDatasetName("");
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    } catch (e) {
      toast({
        title: "Failed to cancel annotation",
        status: "error",
        description: e?.message,
      });
    }
  };

  const onDatasetClick = (ds) => {
    // Use plain ds per your newer snippets; switch to encodeURIComponent if needed.
    navigate(`/data/${ds}`);
  };

  // Enhance prompt (calls backend to refine problemStatement)
  const handleEnhancePrompt = async () => {
    if (!problemStatement.trim()) {
      toast({ title: "Please enter a problem statement", status: "error" });
      return;
    }
    setIsEnhancing(true);
    try {
      const { data } = await jarvisApiClient.post("/api/enhance_prompt", {
        problem_statement: problemStatement.trim(),
      });
      if (!data?.ok || !data?.enhanced_prompt) {
        throw new Error(data?.error || "Failed to enhance prompt");
      }
      setProblemStatement(data.enhanced_prompt);
      toast({ title: "Prompt enhanced", status: "success" });
    } catch (e) {
      toast({
        title: "Failed to enhance prompt",
        status: "error",
        description: e?.response?.data?.error || e?.message || "Unknown error",
      });
    } finally {
      setIsEnhancing(false);
    }
  };

  const handleMonitorUpdate = useCallback((s) => {
    setAnnotationStatus(s);
  }, []);

  const handleMonitorTerminal = useCallback(
    (s) => {
      setIsUploading(false);
      if (s.status === "completed") {
        toast({ title: "Annotation completed 🎉", status: "success" });
        setShowUploader(false);
        setObjectName("");
        setZipFile(null);
        setDatasetName("");
        setConfidenceThreshold(0.8);
        setProblemStatement("");
        setSampleFiles([]);
      } else if (s.status === "failed") {
        toast({ title: "Annotation failed", status: "error" });
      } else if (s.status === "cancelled") {
        toast({ title: "Annotation cancelled", status: "info" });
        setAnnotationStatus(null);
      }
      fetchDatasets();
    },
    [toast, fetchDatasets]
  );

  const UploaderDialog = (
    <Dialog
      open={showUploader}
      onClose={() => (!isUploading ? setShowUploader(false) : null)}
      fullWidth
      maxWidth="md"
      PaperProps={{ sx: { maxHeight: "80vh", overflowY: "auto" } }}
      aria-labelledby="upload-dialog-title"
    >
      <DialogTitle id="upload-dialog-title">
        Add Dataset
        <IconButton
          aria-label="Close dialog"
          onClick={() => (!isUploading ? setShowUploader(false) : null)}
          sx={{ position: "absolute", right: 8, top: 8 }}
          disabled={isUploading}
        >
          <HugeiconsIcon icon={Cancel01Icon} size={24} />
        </IconButton>
      </DialogTitle>
      <DialogContent dividers>
        <Stack component="form" onSubmit={handleUpload} gap={2}>
          <Typography variant="subtitle1" fontWeight={600}>
            Dataset Information
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6}>
              <TextField
                label="Dataset Name (optional)"
                placeholder="e.g., My Car Dataset"
                value={datasetName}
                onChange={(e) => setDatasetName(e.target.value)}
                disabled={isUploading}
                fullWidth
                variant="outlined"
                helperText="Name your dataset for easy reference"
                InputLabelProps={{ shrink: true }}
                aria-describedby="dataset-name-help"
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                label="Object Name(s) *"
                placeholder="e.g., car, person, building (comma-separated)"
                value={objectName}
                onChange={(e) => setObjectName(e.target.value)}
                disabled={isUploading}
                autoFocus
                required
                fullWidth
                variant="outlined"
                helperText="Required: Objects to detect, separated by commas"
                InputLabelProps={{ shrink: true }}
                aria-describedby="object-name-help"
              />
            </Grid>
            <Grid item xs={12}>
              <Stack direction={"column"} gap={1}>
                <TextField
                  label={`Problem Statement ${
                    modelName?.startsWith("reasoning:")
                      ? "(required)"
                      : "(optional)"
                  }`}
                  placeholder="e.g., Detect cars in urban scenes with partial occlusions and varying lighting."
                  value={problemStatement}
                  onChange={(e) => setProblemStatement(e.target.value)}
                  disabled={isUploading || isEnhancing}
                  multiline
                  rows={3}
                  fullWidth
                  variant="outlined"
                  InputLabelProps={{ shrink: true }}
                  aria-describedby="problem-statement-help"
                  sx={{ flexGrow: 1 }}
                  required={modelName?.startsWith("reasoning:")}
                  error={
                    modelName?.startsWith("reasoning:") &&
                    !problemStatement.trim()
                  }
                  helperText={
                    modelName?.startsWith("reasoning:") &&
                    !problemStatement.trim()
                      ? "Problem statement is required for reasoning models"
                      : ""
                  }
                />
                <div className="flex justify-between items-center gap-1 flex-wrap">
                  <FormHelperText>
                    {modelName?.startsWith("reasoning:")
                      ? "Required: Provide detailed context to guide the reasoning model's detection logic."
                      : "Provide context to improve annotations."}
                  </FormHelperText>
                  <Button
                    variant="outlined"
                    onClick={handleEnhancePrompt}
                    disabled={
                      isUploading || isEnhancing || !problemStatement.trim()
                    }
                    size="small"
                    sx={{
                      alignSelf: "flex-end",
                      whiteSpace: "nowrap",
                      minWidth: 180,
                    }}
                  >
                    {isEnhancing ? (
                      <CircularProgress size={20} />
                    ) : (
                      "Enhance Prompt"
                    )}
                  </Button>
                </div>
              </Stack>
            </Grid>
          </Grid>

          <Divider sx={{ my: 2 }} />

          <Typography variant="subtitle1" fontWeight={600}>
            Annotation Settings
          </Typography>
          <Grid item xs={12} sm={6} className="flex-1">
            <Stack gap={1}>
              <Typography variant="subtitle2" fontWeight={500}>
                Confidence Threshold
              </Typography>
              <Stack
                direction="row"
                gap={2}
                alignItems="center"
                width="100%"
                flexWrap="wrap"
              >
                <Slider
                  value={confidenceThreshold}
                  onChange={(_, value) => setConfidenceThreshold(value)}
                  step={0.05}
                  min={0}
                  max={1}
                  disabled={isUploading}
                  valueLabelDisplay="auto"
                  aria-label="Confidence threshold slider"
                  sx={{
                    flexGrow: 1,
                    width: 0,
                    minWidth: 150,
                    "& .MuiSlider-thumb": { bgcolor: "primary.main" },
                    "& .MuiSlider-track": { bgcolor: "primary.light" },
                  }}
                />
                <TextField
                  value={confidenceThreshold}
                  onChange={(e) => {
                    const val = parseFloat(e.target.value);
                    if (!isNaN(val) && val >= 0 && val <= 1) {
                      setConfidenceThreshold(val);
                    }
                  }}
                  type="number"
                  size="small"
                  inputProps={{ step: 0.05, min: 0, max: 1 }}
                  disabled={isUploading}
                  variant="outlined"
                  sx={{ width: 150 }}
                  InputLabelProps={{ shrink: true }}
                />
                <TextField
                  value={imageSize}
                  onChange={(e) => {
                    const val = parseInt(e.target.value);
                    if (!isNaN(val)) {
                      setImageSize(val);
                    }
                  }}
                  type="number"
                  size="small"
                  inputProps={{ step: 1 }}
                  disabled={isUploading}
                  variant="outlined"
                  label="Image Size"
                  sx={{ width: 150 }}
                  InputLabelProps={{ shrink: true }}
                  InputProps={{
                    startAdornment: (
                      <Tooltip
                        title="Enable to upscale the image for higher quality"
                        placement="top"
                      >
                        <InputAdornment position="start">
                          <Checkbox
                            checked={upscaleImage}
                            onChange={(e) => setUpscaleImage(e.target.checked)}
                            disabled={isUploading}
                            size="small"
                          />
                        </InputAdornment>
                      </Tooltip>
                    ),
                    endAdornment: (
                      <Tooltip
                        title="Unit of measurement for image size"
                        placement="top"
                      >
                        <InputAdornment position="end">px</InputAdornment>
                      </Tooltip>
                    ),
                  }}
                />
                <Select
                  value={modelName}
                  onChange={(e) => setModelName(e.target.value)}
                  placeholder="Select model"
                  sx={{ width: 280 }}
                  size="small"
                  disabled={isUploading}
                >
                  {modelOptions.map((option) => (
                    <MenuItem key={option.value} value={option.value}>
                      {option.label}
                    </MenuItem>
                  ))}
                </Select>
              </Stack>
              <Typography variant="caption" color="text.secondary">
                Detections below this score (0-1) are ignored (default: 0.8)
              </Typography>
            </Stack>
          </Grid>

          <Divider sx={{ my: 2 }} />

          <Typography variant="subtitle1" fontWeight={600}>
            Files
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6}>
              <Box
                sx={{
                  border: "2px dashed",
                  borderColor: "divider",
                  borderRadius: 1,
                  p: 2,
                  textAlign: "center",
                  bgcolor: isUploading
                    ? "action.disabledBackground"
                    : "background.paper",
                  "&:hover": {
                    borderColor: isUploading ? "divider" : "primary.main",
                    bgcolor: isUploading
                      ? "action.disabledBackground"
                      : "action.hover",
                  },
                  transition: "all 0.2s",
                }}
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => {
                  if (!isUploading) {
                    e.preventDefault();
                    const file = e.dataTransfer.files?.[0];
                    if (file && file.name.endsWith(".zip")) setZipFile(file);
                    else setUploadError("Please drop a .zip file");
                  }
                }}
              >
                <HugeiconsIcon
                  icon={CloudUploadIcon}
                  size={32}
                  color="action.active"
                />
                <Typography variant="body2" sx={{ mt: 1, mb: 2 }}>
                  Images ZIP File *
                </Typography>
                <input
                  type="file"
                  accept=".zip"
                  ref={fileInputRef}
                  style={{ display: "none" }}
                  onChange={(e) => {
                    const file = e.target.files?.[0] || null;
                    if (file && file.size > 5 * 1024 * 1024 * 1024) {
                      setUploadError("ZIP file too large (max 5GB)");
                    } else {
                      setZipFile(file);
                      setUploadError(null);
                    }
                  }}
                  disabled={isUploading}
                  required
                  id="zip-file-input"
                />
                <Stack
                  direction="row"
                  gap={1}
                  alignItems="center"
                  justifyContent="center"
                >
                  <Button
                    variant="outlined"
                    onClick={() => fileInputRef.current?.click()}
                    disabled={isUploading}
                    size="small"
                  >
                    Choose ZIP
                  </Button>
                  <Chip
                    label={zipFile ? zipFile.name : "No file selected"}
                    onDelete={
                      zipFile && !isUploading
                        ? () => setZipFile(null)
                        : undefined
                    }
                    disabled={isUploading}
                    variant="outlined"
                    size="small"
                  />
                </Stack>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6}>
              <Box
                sx={{
                  border: "2px dashed",
                  borderColor: "divider",
                  borderRadius: 1,
                  p: 2,
                  textAlign: "center",
                  bgcolor: isUploading
                    ? "action.disabledBackground"
                    : "background.paper",
                  "&:hover": {
                    borderColor: isUploading ? "divider" : "primary.main",
                    bgcolor: isUploading
                      ? "action.disabledBackground"
                      : "action.hover",
                  },
                  transition: "all 0.2s",
                }}
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => {
                  if (!isUploading) {
                    e.preventDefault();
                    const files = Array.from(e.dataTransfer.files || [])
                      .filter((f) => f.type.startsWith("image/"))
                      .slice(0, 3);
                    if (files.length === 0) {
                      setUploadError("Please drop valid image files");
                    } else {
                      setSampleFiles(files);
                      setUploadError(null);
                    }
                  }
                }}
              >
                <HugeiconsIcon
                  icon={CloudUploadIcon}
                  size={32}
                  color="action.active"
                />
                <Typography variant="body2" sx={{ mt: 1, mb: 2 }}>
                  Sample Images (up to 3)
                </Typography>
                <input
                  type="file"
                  ref={sampleInputRef}
                  multiple
                  accept="image/*"
                  style={{ display: "none" }}
                  onChange={(e) => {
                    const files = Array.from(e.target.files || [])
                      .filter((f) => f.type.startsWith("image/"))
                      .slice(0, 3);
                    if (files.some((f) => f.size > 5 * 1024 * 1024)) {
                      setUploadError("Sample images must be under 5MB each");
                    } else {
                      setSampleFiles(files);
                      setUploadError(null);
                    }
                  }}
                  disabled={isUploading}
                  id="sample-files-input"
                />
                <Stack
                  direction="row"
                  gap={1}
                  alignItems="center"
                  justifyContent="center"
                  flexWrap="wrap"
                >
                  <Button
                    variant="outlined"
                    onClick={() => sampleInputRef.current?.click()}
                    disabled={isUploading}
                    size="small"
                  >
                    Choose Images
                  </Button>
                  {sampleFiles.length > 0 ? (
                    sampleFiles.map((file, i) => (
                      <Chip
                        key={i}
                        label={file.name}
                        onDelete={
                          !isUploading
                            ? () =>
                                setSampleFiles(
                                  sampleFiles.filter((_, idx) => idx !== i)
                                )
                            : undefined
                        }
                        disabled={isUploading}
                        variant="outlined"
                        size="small"
                      />
                    ))
                  ) : (
                    <Chip
                      label="No files selected"
                      variant="outlined"
                      size="small"
                    />
                  )}
                </Stack>
              </Box>
            </Grid>
          </Grid>

          {uploadError && (
            <Alert severity="error" sx={{ mt: 2 }}>
              {uploadError}
            </Alert>
          )}
          <AnnotationJobMonitor
            active={isUploading}
            pollMs={POLL_MS}
            onUpdate={handleMonitorUpdate}
            onTerminal={handleMonitorTerminal}
          />
        </Stack>
      </DialogContent>
      <DialogActions>
        <Button
          onClick={() => setShowUploader(false)}
          disabled={isUploading}
          color="inherit"
        >
          Close
        </Button>
        <Button
          onClick={handleCancelAnnotation}
          variant="contained"
          color="error"
          disabled={!isUploading}
          startIcon={<HugeiconsIcon icon={Cancel01Icon} />}
        >
          Cancel Annotation
        </Button>
        <Button
          onClick={handleUpload}
          variant="contained"
          disabled={!canSubmit}
          startIcon={<HugeiconsIcon icon={CloudUploadIcon} />}
        >
          {isUploading ? "Processing…" : "Upload & Start"}
        </Button>
      </DialogActions>
    </Dialog>
  );

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Stack
        direction="row"
        alignItems="center"
        justifyContent="space-between"
        mb={3}
      >
        <Typography variant="h5" fontWeight={700}>
          Data Home
        </Typography>
        <Stack direction="row" gap={1}>
          <Tooltip title="Refresh">
            <span>
              <IconButton onClick={fetchDatasets} disabled={isLoading}>
                <HugeiconsIcon icon={RefreshIcon} />
              </IconButton>
            </span>
          </Tooltip>
          <Button
            variant="contained"
            startIcon={<HugeiconsIcon icon={Add01Icon} />}
            onClick={() => setShowUploader(true)}
          >
            Add Dataset
          </Button>
        </Stack>
      </Stack>

      {/* Dataset list */}
      {isLoading ? (
        <Grid container spacing={2}>
          {Array.from({ length: 6 }).map((_, i) => (
            <Grid key={i} item xs={12} sm={6} md={4}>
              <Card>
                <CardContent>
                  <Skeleton variant="text" width="40%" />
                  <Skeleton variant="text" />
                  <Skeleton variant="rectangular" height={24} sx={{ mt: 1 }} />
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      ) : error ? (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      ) : dataSets.length === 0 ? (
        <Box
          sx={{
            border: "1px dashed",
            borderColor: "divider",
            borderRadius: 2,
            p: 6,
            textAlign: "center",
          }}
        >
          <HugeiconsIcon icon={Folder01Icon} />
          <Typography variant="h6" sx={{ mt: 1 }}>
            No datasets yet
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Upload a ZIP of images and we’ll start annotating automatically.
          </Typography>
          <Button
            variant="contained"
            startIcon={<HugeiconsIcon icon={Add01Icon} />}
            onClick={() => setShowUploader(true)}
          >
            Add Dataset
          </Button>
        </Box>
      ) : (
        <>
          <Grid container spacing={2}>
            {dataSets.map((ds) => (
              <Grid key={ds} item xs={12} sm={6} md={4}>
                <Card elevation={1}>
                  <CardActionArea onClick={() => onDatasetClick(ds)}>
                    <CardContent>
                      <Stack
                        direction="row"
                        justifyContent="space-between"
                        alignItems="center"
                      >
                        <Typography variant="subtitle2" color="text.secondary">
                          Dataset
                        </Typography>
                      </Stack>
                      <Typography
                        variant="body1"
                        sx={{
                          mt: 0.5,
                          wordBreak: "break-word",
                        }}
                      >
                        {ds}
                      </Typography>
                      {/*
                      // Re-enable if you want live chip in the list:
                      {annotationStatus && annotationStatus.dataset === ds && (
                        <Stack direction="row" alignItems="center" gap={1} mt={1}>
                          <Chip
                            size="small"
                            label={annotationStatus.status}
                            color={
                              annotationStatus.status === "completed"
                                ? "success"
                                : annotationStatus.status === "failed" ||
                                  annotationStatus.status === "cancelled"
                                ? "error"
                                : "default"
                            }
                          />
                          {annotationStatus.total > 0 && (
                            <Typography variant="caption">
                              ({annotationStatus.current}/{annotationStatus.total})
                            </Typography>
                          )}
                          {annotationStatus.status === "running" && (
                            <Button
                              variant="outlined"
                              color="error"
                              size="small"
                              onClick={handleCancelAnnotation}
                            >
                              Cancel
                            </Button>
                          )}
                        </Stack>
                      )}
                      */}
                    </CardContent>
                  </CardActionArea>
                </Card>
              </Grid>
            ))}
          </Grid>
          <Divider sx={{ my: 3 }} />
          <Typography variant="caption" color="text.secondary">
            Tip: Click a card to view details and annotations.
          </Typography>
        </>
      )}

      {UploaderDialog}
    </Container>
  );
};

export default DataHome;
