import React, {
  useEffect,
  useState,
  useCallback,
  useRef,
  useMemo,
} from "react";
import { useNavigate } from "react-router-dom";
import {
  Container,
  Stack,
  Typography,
  Button,
  Card,
  CardContent,
  IconButton,
  Tooltip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Alert,
  CircularProgress,
  Grid,
  Chip,
  Select,
  MenuItem,
  Box,
  FormControl,
  InputLabel,
} from "@mui/material";
import {
  Add01Icon,
  RefreshIcon,
  Download01Icon,
  Delete02Icon,
  Folder01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { jarvisApiClient } from "../../../api/api";
import { useToast } from "../../Toast/Toast";

const POLL_MS = 2000;

const ModelHome = () => {
  const navigate = useNavigate();
  const toast = useToast();

  // interval + notifications refs
  const pollRef = useRef(null);
  const notifiedRef = useRef({}); // dataset -> last terminal status already notified

  // State for models
  const [models, setModels] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  // State for fine-tuning dialog
  const [showFineTuneDialog, setShowFineTuneDialog] = useState(false);
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState("");
  const [epochs, setEpochs] = useState(100);
  const [batchSize, setBatchSize] = useState(16);
  const [isTraining, setIsTraining] = useState(false);

  // dataset|timestamp -> { status, message }
  const [trainingStatus, setTrainingStatus] = useState({});

  const fetchModels = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);
      const { data } = await jarvisApiClient.get("/api/models");
      if (!data?.ok) throw new Error(data?.error || "Failed to load models");
      setModels(Array.isArray(data.models) ? data.models : []);
    } catch (e) {
      setError(
        e?.response?.data?.error || e?.message || "Failed to load models"
      );
    } finally {
      setIsLoading(false);
    }
  }, []);

  const fetchDatasets = useCallback(async () => {
    try {
      const { data } = await jarvisApiClient.get("/api/datasets");
      if (!data?.ok) throw new Error(data?.error || "Failed to load datasets");
      setDatasets(Array.isArray(data.datasets) ? data.datasets : []);
    } catch (e) {
      toast({
        title: "Failed to load datasets",
        status: "error",
        description: e?.message,
      });
    }
  }, [toast]);

  const clearPoller = () => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  };

  const startPollingTraining = useCallback(() => {
    // Avoid multiple intervals
    if (pollRef.current) return;

    pollRef.current = setInterval(async () => {
      try {
        const { data } = await jarvisApiClient.get("/api/train/status");
        if (!data?.ok || data.status === "idle") {
          // NEW: Clear trainingStatus on idle and stop polling
          setTrainingStatus({});
          clearPoller();
          return;
        }

        // We assume API returns a single current job: { dataset, status, message }
        const ds = data.dataset;
        const status = data.status;

        setTrainingStatus((prev) => ({
          ...prev,
          [ds]: { status, message: data.message },
        }));

        const isTerminal =
          status === "completed" ||
          status === "failed" ||
          status === "cancelled";
        if (isTerminal) {
          const alreadyNotified = notifiedRef.current[ds] === status;
          if (!alreadyNotified) {
            notifiedRef.current[ds] = status;
            toast({
              title:
                status === "completed"
                  ? "Training completed"
                  : status === "failed"
                  ? "Training failed"
                  : "Training cancelled",
              status: status === "completed" ? "success" : "error",
              description: data.message,
            });

            // Update model list after terminal state
            fetchModels();

            if (status === "failed" || status === "cancelled") {
              // Best-effort cleanup
              try {
                await jarvisApiClient.post(`/api/train/${ds}/cleanup`);
              } catch {}
              setTrainingStatus((prev) => {
                const next = { ...prev };
                delete next[ds];
                return next;
              });
            }
          }

          // Stop polling once a job reaches a terminal state (assuming one-at-a-time)
          clearPoller();
        }
      } catch {
        // ignore transient poll errors
      }
    }, POLL_MS);
  }, [fetchModels, toast]);

  useEffect(() => {
    fetchModels();
    fetchDatasets();
    startPollingTraining(); // in case a job is already running

    return clearPoller;
  }, [fetchModels, fetchDatasets, startPollingTraining]);

  const handleFineTune = async () => {
    if (!selectedDataset) {
      toast({ title: "Please select a dataset", status: "error" });
      return;
    }
    if (epochs <= 0 || batchSize <= 0) {
      toast({
        title: "Epochs and batch size must be positive",
        status: "error",
      });
      return;
    }

    setIsTraining(true);
    try {
      const { data } = await jarvisApiClient.post("/api/train", {
        dataset: selectedDataset,
        epochs,
        batch_size: batchSize,
      });
      if (!data?.ok) throw new Error(data?.error || "Failed to start training");

      setTrainingStatus((prev) => ({
        ...prev,
        [selectedDataset]: { status: "running", message: "Training started" },
      }));
      delete notifiedRef.current[selectedDataset];

      toast({ title: "Training started", status: "info" });
      setShowFineTuneDialog(false);
      startPollingTraining();
    } catch (e) {
      toast({
        title: "Failed to start training",
        status: "error",
        description: e?.message,
      });
    } finally {
      setIsTraining(false);
    }
  };

  // NEW: allow cancelling current training
  const handleCancelTraining = async (datasetKey) => {
    try {
      const { data } = await jarvisApiClient.post("/api/train/cancel");
      if (!data?.ok)
        throw new Error(data?.error || "Failed to cancel training");
      toast({
        title: "Training cancelled",
        status: "info",
        description: data.message,
      });
      setTrainingStatus((prev) => {
        const next = { ...prev };
        delete next[datasetKey];
        return next;
      });
      clearPoller();
    } catch (e) {
      toast({
        title: "Failed to cancel training",
        status: "error",
        description: e?.message,
      });
    }
  };

  const handleTestInference = (model) => {
    navigate(`/model/inference/${model.timestamp}`);
  };

  const handleDownload = (model) => {
    window.open(`/api/models/${model.timestamp}/download`, "_blank");
  };

  const handleDelete = async (model) => {
    if (window.confirm(`Delete model ${model.timestamp}?`)) {
      try {
        const { data } = await jarvisApiClient.delete(
          `/api/models/${model.timestamp}`
        );
        if (!data?.ok) throw new Error(data?.error || "Failed to delete model");
        toast({ title: "Model deleted", status: "success" });
        fetchModels();
      } catch (e) {
        toast({
          title: "Failed to delete model",
          status: "error",
          description: e?.message,
        });
      }
    }
  };

  // safer key for looking up status on cards
  const statusKeyForModel = useCallback(
    (model) => model.dataset ?? model.timestamp,
    []
  );

  const selectedStatus = trainingStatus[selectedDataset];

  // Build a set of "existing" identifiers to avoid duplicating pending cards
  const existingKeys = useMemo(() => {
    const s = new Set();
    for (const m of models) {
      if (m.timestamp) s.add(m.timestamp);
      if (m.dataset) s.add(m.dataset);
    }
    return s;
  }, [models]);

  const hasAnyTraining = Object.keys(trainingStatus).length > 0;

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Stack
        direction="row"
        alignItems="center"
        justifyContent="space-between"
        mb={3}
      >
        <Typography variant="h5" fontWeight={700}>
          Model Home
        </Typography>
        <Stack direction="row" gap={1}>
          <Tooltip title="Refresh">
            <span>
              <IconButton onClick={fetchModels} disabled={isLoading}>
                <HugeiconsIcon icon={RefreshIcon} />
              </IconButton>
            </span>
          </Tooltip>
          <Button
            variant="contained"
            startIcon={<HugeiconsIcon icon={Add01Icon} />}
            onClick={() => setShowFineTuneDialog(true)}
          >
            Fine-Tune Model
          </Button>
        </Stack>
      </Stack>

      {isLoading ? (
        <CircularProgress />
      ) : error ? (
        <Alert severity="error">{error}</Alert>
      ) : models.length === 0 && !hasAnyTraining ? (
        <Box sx={{ textAlign: "center", p: 4 }}>
          <HugeiconsIcon icon={Folder01Icon} size={48} />
          <Typography variant="h6" mt={2}>
            No models yet
          </Typography>
          <Typography variant="body2" color="text.secondary" mb={2}>
            Fine-tune a model from an annotated dataset.
          </Typography>
          <Button
            variant="contained"
            onClick={() => setShowFineTuneDialog(true)}
          >
            Start Fine-Tuning
          </Button>
        </Box>
      ) : (
        <Grid container spacing={2}>
          {/* Pending/in-progress cards (show while not yet materialized in the models list) */}
          {Object.keys(trainingStatus).map((key) => {
            const ts = trainingStatus[key];
            if (ts.status !== "running") return null;
            if (existingKeys.has(key)) return null; // avoid dup if a model already exists with this key
            return (
              <Grid key={`pending-${key}`} item xs={12} sm={6} md={4}>
                <Card>
                  <CardContent>
                    <Typography variant="subtitle1">Training {key}</Typography>
                    <Typography variant="body2" color="text.secondary">
                      Identifier: {key}
                    </Typography>
                    <Chip label={ts.status} color="default" sx={{ mt: 1 }} />
                    {ts.message ? (
                      <Typography variant="body2" sx={{ mt: 1 }}>
                        {ts.message}
                      </Typography>
                    ) : null}
                    <Stack direction="row" gap={1} mt={2}>
                      <Button
                        variant="outlined"
                        color="error"
                        onClick={() => handleCancelTraining(key)}
                      >
                        Cancel Training
                      </Button>
                    </Stack>
                  </CardContent>
                </Card>
              </Grid>
            );
          })}

          {/* Saved models */}
          {models.map((model) => {
            const skey = statusKeyForModel(model);
            const statusObj = trainingStatus[skey];
            const color =
              statusObj?.status === "completed"
                ? "success"
                : statusObj?.status === "failed"
                ? "error"
                : statusObj?.status === "cancelled"
                ? "error"
                : "default";
            return (
              <Grid key={model.timestamp} item xs={12} sm={6} md={4}>
                <Card>
                  <CardContent>
                    <Typography variant="subtitle1">{model.name}</Typography>
                    <Typography variant="body2" color="text.secondary">
                      Timestamp: {model.timestamp}
                    </Typography>
                    {statusObj?.status && (
                      <Chip
                        label={statusObj.status}
                        color={color}
                        sx={{ mt: 1 }}
                      />
                    )}
                    <Stack direction="row" gap={1} mt={2}>
                      <Button
                        variant="outlined"
                        onClick={() => handleTestInference(model)}
                      >
                        Test Inference
                      </Button>
                      <IconButton onClick={() => handleDownload(model)}>
                        <HugeiconsIcon icon={Download01Icon} />
                      </IconButton>
                      <IconButton
                        color="error"
                        onClick={() => handleDelete(model)}
                      >
                        <HugeiconsIcon icon={Delete02Icon} />
                      </IconButton>
                    </Stack>
                  </CardContent>
                </Card>
              </Grid>
            );
          })}
        </Grid>
      )}

      {/* Fine-Tune Dialog */}
      <Dialog
        open={showFineTuneDialog}
        onClose={() => setShowFineTuneDialog(false)}
        fullWidth
        maxWidth="sm"
      >
        <DialogTitle>Fine-Tune Model</DialogTitle>
        <DialogContent>
          <Stack gap={2} mt={1}>
            <FormControl fullWidth>
              <InputLabel id="dataset-select-label">Select Dataset</InputLabel>
              <Select
                labelId="dataset-select-label"
                value={selectedDataset}
                label="Select Dataset"
                onChange={(e) => setSelectedDataset(e.target.value)}
                disabled={isTraining}
              >
                {datasets.map((ds) => (
                  <MenuItem key={ds} value={ds}>
                    {ds}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            <TextField
              label="Epochs"
              type="number"
              value={epochs}
              onChange={(e) => setEpochs(Math.max(1, Number(e.target.value)))}
              fullWidth
              disabled={isTraining}
              inputProps={{ min: 1 }}
            />
            <TextField
              label="Batch Size"
              type="number"
              value={batchSize}
              onChange={(e) =>
                setBatchSize(Math.max(1, Number(e.target.value)))
              }
              fullWidth
              disabled={isTraining}
              inputProps={{ min: 1 }}
            />
          </Stack>

          {isTraining && (
            <Stack direction="row" alignItems="center" gap={1} sx={{ mt: 2 }}>
              <CircularProgress size={20} />
              <Typography variant="body2">Starting training…</Typography>
            </Stack>
          )}

          {selectedDataset && selectedStatus?.status && (
            <Typography sx={{ mt: 1 }}>
              Status: {selectedStatus.status}
              {selectedStatus.message ? ` — ${selectedStatus.message}` : ""}
            </Typography>
          )}
        </DialogContent>
        <DialogActions>
          <Button
            onClick={() => setShowFineTuneDialog(false)}
            disabled={isTraining}
          >
            Cancel
          </Button>
          <Button
            variant="contained"
            color="error"
            onClick={() => handleCancelTraining(selectedDataset)}
            disabled={!trainingStatus[selectedDataset]?.status}
          >
            Cancel Training
          </Button>
          <Button
            onClick={handleFineTune}
            variant="contained"
            disabled={isTraining || !selectedDataset}
          >
            Start Fine-Tuning
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
};

export default ModelHome;
