import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart'
    show rootBundle; // For loading TFLite model from assets
import 'package:http/http.dart' as http;
import 'package:image/image.dart' as img_lib;
import 'package:tflite_flutter/tflite_flutter.dart';

// --- Configuration ---
const String defaultPythonServerPort =
    "5000"; // Port MobileAppHttpController.py runs on
const String tfliteModelFileName =
    "detect.tflite"; // Your TFLite model in assets/

// Common COCO dataset class names.
// IMPORTANT: This map MUST match the output of your TFLite model.
// If your model is trained on COCO with 80 classes, ensure this map covers indices 0-79.
// Your previous map was quite extensive, this is a slightly more concise version.
// Adjust if your model has different classes or a different number of classes.
const Map<int, String> classNames = {
  0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
  5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
  10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
  14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
  20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
  25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
  30: 'skis',
  31: 'snowboard',
  32: 'sports ball',
  33: 'kite',
  34: 'baseball bat',
  35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
  39: 'bottle',
  40: 'wine glass',
  41: 'cup',
  42: 'fork',
  43: 'knife',
  44: 'spoon',
  45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
  50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
  55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
  60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',
  65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
  69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
  74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
  79: 'toothbrush',
  // Add more if your model has more classes.
};

class DetectedObject {
  final Rect boundingBox;
  final int classId;
  final double confidence;
  final String className;

  DetectedObject({
    required this.boundingBox,
    required this.classId,
    required this.confidence,
  }) : className =
           classNames[classId] ??
           'ClassID ${classId}'; // Fallback for unknown class IDs

  @override
  String toString() =>
      'Object: $className (${(confidence * 100).toStringAsFixed(1)}%) Box: $boundingBox';
}

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const CarControlApp());
}

class CarControlApp extends StatelessWidget {
  const CarControlApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Car Controller & TFLite',
      theme: ThemeData(
        brightness: Brightness.dark,
        primarySwatch: Colors.teal,
        primaryColor: Colors.teal[700],
        scaffoldBackgroundColor: const Color(0xFF22272B), // Darker background
        cardColor: const Color(0xFF2D3339), // Card color
        textTheme: Typography.whiteMountainView.apply(fontFamily: 'Roboto'),
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            backgroundColor: Colors.teal[600],
            foregroundColor: Colors.white,
            padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 15),
            textStyle: const TextStyle(
              fontSize: 16,
              fontWeight: FontWeight.w500,
            ),
          ),
        ),
        inputDecorationTheme: InputDecorationTheme(
          border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
          filled: true,
          fillColor: const Color(0xFF373E47),
        ),
        appBarTheme: AppBarTheme(
          backgroundColor: Colors.teal[800],
          elevation: 4,
        ),
        useMaterial3: true,
      ),
      home: const ControlPage(),
      debugShowCheckedModeBanner: false,
    );
  }
}

class ControlPage extends StatefulWidget {
  const ControlPage({super.key});

  @override
  State<ControlPage> createState() => _ControlPageState();
}

class _ControlPageState extends State<ControlPage> {
  final TextEditingController _ipController = TextEditingController();
  String _serverBaseUrl = '';
  String _videoStreamUrl = '';
  String _statusApiUrl = '';
  String _controlApiBaseUrl = '';

  String _currentStatusMessage = 'Disconnected. Load TFLite Model first.';
  bool _isConnected = false;
  bool _isTfliteProcessing = false;
  bool _isObjectDetectionEnabled = false;
  bool _isVideoLoading = false; // For initial video load indicator

  // TFLite
  Interpreter? _tfliteInterpreter;
  bool _isTfliteModelLoaded = false;
  int? _tfliteInputHeight;
  int? _tfliteInputWidth;
  int _tfliteMaxDetections = 10; // Default, updated from model
  int _tfBoxesIdx = -1,
      _tfClassesIdx = -1,
      _tfScoresIdx = -1,
      _tfNumDetectionsIdx = -1; // TFLite output indices

  List<DetectedObject> _detectedObjects = [];
  Size? _originalImageSizeForPainter; // For scaling bounding boxes

  // Video Stream
  http.Client? _videoHttpClient;
  StreamSubscription? _videoStreamSubscription;
  Uint8List? _currentFrameBytes;
  Timer? _frameDisplayTimer; // To update UI at a reasonable rate
  Timer? _tfliteThrottleTimer; // To limit TFLite processing frequency
  late Uint8List _mjpegBoundaryBytes;

  // Performance counters
  int _framesReceivedCounter = 0;
  int _tfliteRunsCounter = 0;
  DateTime? _lastPerfUpdateTime;
  String _perfStats = "FPS: 0, Detections/s: 0";

  // Status from Python server
  Map<String, dynamic> _serverStatus = {};

  @override
  void initState() {
    super.initState();
    _ipController.text = '192.168.1.100'; // Default IP, change if needed
    _mjpegBoundaryBytes = utf8.encode(
      '--frame\r\nContent-Type: image/jpeg\r\n\r\n',
    );
    _initializeTflite();
  }

  Future<void> _initializeTflite() async {
    final String modelPath = 'assets/$tfliteModelFileName';
    try {
      // Verify asset exists (optional but good for debugging)
      await rootBundle.load(modelPath);
      debugPrint("TFLite: Model asset '$modelPath' found.");

      _tfliteInterpreter = await Interpreter.fromAsset(modelPath);
      _tfliteInterpreter!.allocateTensors(); // Crucial step

      final inputTensor = _tfliteInterpreter!.getInputTensor(0);
      _tfliteInputHeight = inputTensor.shape[1]; // e.g., 320
      _tfliteInputWidth = inputTensor.shape[2]; // e.g., 320

      debugPrint(
        "TFLite: Model Input Shape: ${inputTensor.shape}, Type: ${inputTensor.type}",
      );

      // Dynamically determine output tensor indices and _tfliteMaxDetections
      // This matches the logic from your previous Flutter code for parsing model outputs.
      int outputTensorCount = _tfliteInterpreter!.getOutputTensors().length;
      debugPrint("TFLite: Number of output tensors: $outputTensorCount");

      bool mappedAllTensors = false;
      // Attempt mapping by common TFLite SSD MobileNet output tensor names/suffixes
      for (int i = 0; i < outputTensorCount; i++) {
        var tensor = _tfliteInterpreter!.getOutputTensor(i);
        debugPrint(
          "  Output $i: Name=${tensor.name}, Shape=${tensor.shape}, Type=${tensor.type}",
        );

        // These suffixes are common but might need adjustment based on your exact model conversion
        if (tensor.name.endsWith('TFLite_Detection_PostProcess') ||
            tensor.name.endsWith('detection_boxes') ||
            tensor.name.endsWith(':3')) {
          // Boxes
          if (tensor.shape.length == 3 &&
              tensor.shape[0] == 1 &&
              tensor.shape[2] == 4) {
            _tfBoxesIdx = i;
            _tfliteMaxDetections = tensor.shape[1];
          }
        } else if (tensor.name.endsWith('TFLite_Detection_PostProcess:1') ||
            tensor.name.endsWith('detection_classes') ||
            tensor.name.endsWith(':2')) {
          // Classes
          if (tensor.shape.length == 2 && tensor.shape[0] == 1) {
            _tfClassesIdx = i;
            if (_tfliteMaxDetections == 10 ||
                (_tfliteMaxDetections != tensor.shape[1] &&
                    tensor.shape[1] > 0))
              _tfliteMaxDetections = tensor.shape[1];
          }
        } else if (tensor.name.endsWith('TFLite_Detection_PostProcess:2') ||
            tensor.name.endsWith('detection_scores') ||
            tensor.name.endsWith(':1')) {
          // Scores
          if (tensor.shape.length == 2 && tensor.shape[0] == 1) {
            _tfScoresIdx = i;
            if (_tfliteMaxDetections == 10 ||
                (_tfliteMaxDetections != tensor.shape[1] &&
                    tensor.shape[1] > 0))
              _tfliteMaxDetections = tensor.shape[1];
          }
        } else if (tensor.name.endsWith('TFLite_Detection_PostProcess:3') ||
            tensor.name.endsWith('num_detections') ||
            tensor.name.endsWith(':0')) {
          // Num Detections
          if ((tensor.shape.length == 1 && tensor.shape[0] == 1) ||
              (tensor.shape.length == 2 &&
                  tensor.shape[0] == 1 &&
                  tensor.shape[1] == 1)) {
            _tfNumDetectionsIdx = i;
          }
        }
      }

      if (_tfBoxesIdx != -1 &&
          _tfClassesIdx != -1 &&
          _tfScoresIdx != -1 &&
          _tfNumDetectionsIdx != -1) {
        debugPrint(
          "TFLite: Successfully mapped all output tensors by name convention.",
        );
        mappedAllTensors = true;
      } else {
        // Fallback to shape-based mapping (less reliable)
        debugPrint(
          "TFLite: Name-based mapping failed. Attempting shape-based (less reliable).",
        );
        _tfBoxesIdx = -1;
        _tfClassesIdx = -1;
        _tfScoresIdx = -1;
        _tfNumDetectionsIdx = -1;
        _tfliteMaxDetections = 10;
        List<int> twoDimTensorIndices = [];
        for (int i = 0; i < outputTensorCount; i++) {
          var tensor = _tfliteInterpreter!.getOutputTensor(i);
          if (tensor.shape.length == 3 &&
              tensor.shape[0] == 1 &&
              tensor.shape[2] == 4) {
            // Boxes [1, N, 4]
            _tfBoxesIdx = i;
            _tfliteMaxDetections = tensor.shape[1];
          } else if ((tensor.shape.length == 1 && tensor.shape[0] == 1) ||
              (tensor.shape.length == 2 &&
                  tensor.shape[0] == 1 &&
                  tensor.shape[1] == 1)) {
            // Num Detections [1] or [1,1]
            _tfNumDetectionsIdx = i;
          } else if (tensor.shape.length == 2 && tensor.shape[0] == 1) {
            // Potential classes or scores [1, N]
            twoDimTensorIndices.add(i);
            if (_tfliteMaxDetections == 10 ||
                (_tfliteMaxDetections != tensor.shape[1] &&
                    tensor.shape[1] > 0))
              _tfliteMaxDetections = tensor.shape[1];
          }
        }
        if (twoDimTensorIndices.length >= 2) {
          // Need at least two [1,N] for scores and classes
          // This is a common order, but highly dependent on model conversion
          _tfScoresIdx = twoDimTensorIndices[0]; // Assume first is scores
          _tfClassesIdx = twoDimTensorIndices[1]; // Assume second is classes
        }
        if (_tfBoxesIdx != -1 &&
            _tfClassesIdx != -1 &&
            _tfScoresIdx != -1 &&
            _tfNumDetectionsIdx != -1) {
          mappedAllTensors = true;
        }
      }

      if (!mappedAllTensors) {
        throw Exception(
          "Failed to map all TFLite output tensors. Check model structure and tensor names/order. Indices found: B:$_tfBoxesIdx, C:$_tfClassesIdx, S:$_tfScoresIdx, N:$_tfNumDetectionsIdx",
        );
      }
      debugPrint(
        "TFLite: Output Tensor Indices: Boxes: $_tfBoxesIdx, Classes: $_tfClassesIdx, Scores: $_tfScoresIdx, NumDetections: $_tfNumDetectionsIdx. Max Detections: $_tfliteMaxDetections",
      );

      if (_tfliteInputHeight == null ||
          _tfliteInputWidth == null ||
          _tfliteInputHeight! <= 0 ||
          _tfliteInputWidth! <= 0) {
        throw Exception(
          "Invalid TFLite input dimensions derived from model: H:$_tfliteInputHeight, W:$_tfliteInputWidth",
        );
      }

      if (mounted) {
        setState(() {
          _isTfliteModelLoaded = true;
          _currentStatusMessage =
              "TFLite Model Loaded. Input: [1, $_tfliteInputHeight, $_tfliteInputWidth, 3]. Max Det: $_tfliteMaxDetections. Ready.";
        });
      }
      debugPrint("TFLite: Interpreter initialized successfully.");
    } catch (e, stackTrace) {
      debugPrint("TFLite: Error initializing interpreter: $e");
      debugPrint("TFLite: StackTrace: $stackTrace");
      if (mounted) {
        setState(() {
          _currentStatusMessage =
              "Error TFLite Model: ${e.toString().replaceAll("Exception: ", "")}";
          _isTfliteModelLoaded = false;
        });
      }
    }
  }

  Future<void> _connectToServer() async {
    if (!_isTfliteModelLoaded) {
      _showSnackBar('TFLite model not loaded. Cannot connect.');
      await _initializeTflite(); // Try to reload
      if (!_isTfliteModelLoaded && mounted) {
        _showSnackBar('TFLite Model still not loaded. Aborting connect.');
        return;
      }
    }

    final String ipAddress = _ipController.text.trim();
    if (ipAddress.isEmpty) {
      _showSnackBar('Please enter Python Server IP Address');
      return;
    }

    _disconnectFromServer(); // Clean up previous connection if any

    setState(() {
      _serverBaseUrl = 'http://$ipAddress:$defaultPythonServerPort';
      _videoStreamUrl = '$_serverBaseUrl/video_feed';
      _statusApiUrl = '$_serverBaseUrl/api/status';
      _controlApiBaseUrl = '$_serverBaseUrl/api/control';

      _currentStatusMessage = 'Connecting to $_serverBaseUrl...';
      _isVideoLoading = true;
      _currentFrameBytes = null;
      _detectedObjects.clear();
      _framesReceivedCounter = 0;
      _tfliteRunsCounter = 0;
      _lastPerfUpdateTime = DateTime.now();
      _serverStatus.clear();
    });

    try {
      // Test basic connection by fetching status
      final response = await http
          .get(Uri.parse(_statusApiUrl))
          .timeout(const Duration(seconds: 8));
      if (response.statusCode == 200) {
        debugPrint('Python Server connected. Status: ${response.body}');
        if (mounted) {
          setState(() {
            _isConnected = true;
            _originalImageSizeForPainter = null; // Reset for new stream
            _currentStatusMessage = 'Connected. Starting video...';
            _serverStatus = jsonDecode(response.body);
          });
        }
        await _startVideoStream();
        _startStatusPolling();
      } else {
        throw Exception('Server returned status: ${response.statusCode}');
      }
    } catch (e) {
      debugPrint('Connection to Python Server failed: $e');
      if (mounted) {
        setState(() {
          _currentStatusMessage =
              'Connection failed: ${e.toString().replaceAll('Exception: ', '')}';
          _isConnected = false;
          _isVideoLoading = false;
        });
      }
    }
  }

  Future<void> _startVideoStream() async {
    if (_videoStreamUrl.isEmpty || !_isConnected) return;
    debugPrint('Starting video stream from: $_videoStreamUrl');

    try {
      _videoHttpClient = http.Client();
      final request = http.Request('GET', Uri.parse(_videoStreamUrl));
      // Headers for MJPEG stream
      request.headers.addAll({
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Accept': 'multipart/x-mixed-replace',
      });
      final streamedResponse = await _videoHttpClient!
          .send(request)
          .timeout(const Duration(seconds: 10));

      if (streamedResponse.statusCode != 200) {
        throw Exception(
          'Video stream error: ${streamedResponse.statusCode} ${streamedResponse.reasonPhrase}',
        );
      }
      debugPrint('Video stream HTTP connection successful.');

      List<int> frameBuffer =
          []; // Temporary buffer for assembling a single frame
      _videoStreamSubscription = streamedResponse.stream.listen(
        (List<int> chunk) {
          if (!mounted || !_isConnected) return;
          frameBuffer.addAll(chunk);
          _processMJPEGBuffer(
            frameBuffer,
          ); // Pass the buffer to be potentially modified
        },
        onError: (error, stackTrace) {
          debugPrint('Video Stream Error: $error\n$stackTrace');
          if (mounted) {
            setState(() {
              _currentStatusMessage = 'Video stream error: $error';
              _isVideoLoading = false;
            });
          }
          _stopVideoStream();
        },
        onDone: () {
          debugPrint('Video Stream Ended by Server.');
          if (mounted) {
            setState(() {
              _currentStatusMessage = 'Video stream ended.';
              _isVideoLoading = false;
            });
          }
          _stopVideoStream();
        },
        cancelOnError: true,
      );

      // UI update timer for smoother frame display
      _frameDisplayTimer = Timer.periodic(const Duration(milliseconds: 40), (
        timer,
      ) {
        // ~25 FPS UI updates
        if (mounted && _currentFrameBytes != null) {
          setState(() {}); // Trigger rebuild to display new frame
          _updatePerformanceStats();
        }
      });

      if (mounted) {
        setState(() {
          _currentStatusMessage = 'Video stream active.';
          _isVideoLoading = false;
        });
      }
    } catch (e) {
      debugPrint('Failed to start video stream: $e');
      if (mounted) {
        setState(() {
          _currentStatusMessage =
              'Video stream failed: ${e.toString().replaceAll('Exception: ', '')}';
          _isVideoLoading = false;
        });
      }
      _stopVideoStream();
    }
  }

  void _processMJPEGBuffer(List<int> buffer) {
    if (!mounted || !_isConnected) return;
    try {
      // Continuously process buffer as long as full frames can be extracted
      while (buffer.isNotEmpty) {
        // Find the start of an MJPEG frame
        int boundaryIndex = _indexOfBytes(buffer, _mjpegBoundaryBytes, 0);
        if (boundaryIndex == -1)
          return; // Incomplete frame part, wait for more data

        int jpegStartIndex = boundaryIndex + _mjpegBoundaryBytes.length;
        // Find the start of the NEXT MJPEG frame to delimit the current one
        int nextBoundaryIndex = _indexOfBytes(
          buffer,
          _mjpegBoundaryBytes,
          jpegStartIndex,
        );

        if (nextBoundaryIndex != -1) {
          // Full JPEG frame data is between jpegStartIndex and nextBoundaryIndex
          Uint8List jpegData = Uint8List.fromList(
            buffer.sublist(jpegStartIndex, nextBoundaryIndex),
          );

          if (jpegData.isNotEmpty && jpegData.length > 500) {
            // Basic validation
            _currentFrameBytes = jpegData; // Store for UI update by timer
            _framesReceivedCounter++;

            if (_isObjectDetectionEnabled &&
                _isTfliteModelLoaded &&
                !_isTfliteProcessing &&
                _shouldRunTfliteNow()) {
              _runTfliteObjectDetection(jpegData); // Process a copy
            }
          }
          // Remove processed frame (including its boundary) from the buffer
          buffer.removeRange(0, nextBoundaryIndex);
        } else {
          return; // Current frame is incomplete, wait for more data
        }
      }
    } catch (e, stackTrace) {
      debugPrint('Error processing MJPEG buffer: $e\n$stackTrace');
    }

    // Buffer management: Prevent excessive memory usage if something goes wrong
    if (buffer.length > 3 * 1024 * 1024) {
      // 3MB limit
      debugPrint(
        "MJPEG buffer > 3MB, attempting to trim to last boundary or clear...",
      );
      int lastBoundary = _lastIndexOfBytes(buffer, _mjpegBoundaryBytes);
      if (lastBoundary != -1 && lastBoundary > 0) {
        buffer.removeRange(0, lastBoundary);
        debugPrint(
          "MJPEG buffer trimmed to ${buffer.length} bytes starting from last boundary.",
        );
      } else {
        buffer.clear(); // Clear if no boundary found in large buffer
        debugPrint(
          "MJPEG buffer cleared due to excessive size without recognizable boundary.",
        );
      }
    }
  }

  bool _shouldRunTfliteNow() {
    // Allow TFLite processing if throttle timer is not active
    return _tfliteThrottleTimer?.isActive != true;
  }

  void _stopVideoStream() {
    _videoStreamSubscription?.cancel();
    _videoStreamSubscription = null;
    _videoHttpClient?.close(); // Important to close the client
    _videoHttpClient = null;
    _frameDisplayTimer?.cancel();
    _frameDisplayTimer = null;
    _tfliteThrottleTimer?.cancel();
    _tfliteThrottleTimer = null;
    debugPrint("Video stream and related timers stopped.");
  }

  void _disconnectFromServer() {
    _stopVideoStream();
    _stopStatusPolling();
    if (mounted) {
      setState(() {
        _isConnected = false;
        _videoStreamUrl = '';
        _serverBaseUrl = '';
        _currentStatusMessage = 'Disconnected. TFLite Model still loaded.';
        _detectedObjects.clear();
        _originalImageSizeForPainter = null;
        _isVideoLoading = false;
        _currentFrameBytes = null; // Clear last frame
        _serverStatus.clear();
      });
    }
  }

  void _toggleObjectDetection() {
    if (!_isTfliteModelLoaded) {
      _showSnackBar('TFLite model not initialized!');
      return;
    }
    setState(() {
      _isObjectDetectionEnabled = !_isObjectDetectionEnabled;
      if (!_isObjectDetectionEnabled) {
        _detectedObjects.clear(); // Clear boxes when disabling
      }
      _currentStatusMessage = _isObjectDetectionEnabled
          ? "TFLite Detection ON"
          : "TFLite Detection OFF";
    });
  }

  Future<void> _sendControlCommand(String direction) async {
    if (!_isConnected || _controlApiBaseUrl.isEmpty) {
      if (mounted)
        setState(() => _currentStatusMessage = 'Not connected to server');
      return;
    }
    final url = Uri.parse('$_controlApiBaseUrl/$direction');
    try {
      final response = await http.post(url).timeout(const Duration(seconds: 2));
      if (response.statusCode == 200) {
        final responseData = jsonDecode(response.body);
        debugPrint('Control: $direction. Server: ${responseData['message']}');
        if (mounted) {
          // setState(() => _currentStatusMessage = 'CMD: $direction OK'); // Can be too chatty
        }
      } else {
        debugPrint('Control command $direction failed: ${response.statusCode}');
        if (mounted)
          setState(
            () =>
                _currentStatusMessage = 'Control Error: ${response.statusCode}',
          );
      }
    } catch (e) {
      debugPrint('Control command $direction exception: $e');
      if (mounted)
        setState(
          () => _currentStatusMessage =
              'Control Error: ${e.toString().split(':').last.trim()}',
        );
    }
  }

  Future<void> _runTfliteObjectDetection(Uint8List imageBytes) async {
    if (_tfliteInterpreter == null ||
        !_isTfliteModelLoaded ||
        _isTfliteProcessing ||
        !mounted ||
        !_isObjectDetectionEnabled ||
        _tfliteInputHeight == null ||
        _tfliteInputWidth == null ||
        _tfBoxesIdx == -1 ||
        _tfClassesIdx == -1 ||
        _tfScoresIdx == -1 ||
        _tfNumDetectionsIdx == -1) {
      return;
    }

    _isTfliteProcessing = true;
    _tfliteRunsCounter++;

    // Throttle TFLite runs to manage performance (e.g., max 5-10 FPS for detection)
    _tfliteThrottleTimer = Timer(
      const Duration(milliseconds: 150),
      () {},
    ); // ~6-7 FPS processing

    try {
      final stopwatch = Stopwatch()..start();

      final img_lib.Image? decodedImage = img_lib.decodeImage(
        imageBytes,
      ); // Use generic decodeImage
      if (decodedImage == null) {
        debugPrint("TFLite: Failed to decode image for processing.");
        _isTfliteProcessing = false;
        return;
      }

      final int originalWidth = decodedImage.width;
      final int originalHeight = decodedImage.height;

      final img_lib.Image resizedImage = img_lib.copyResize(
        decodedImage,
        width: _tfliteInputWidth!,
        height: _tfliteInputHeight!,
        interpolation:
            img_lib.Interpolation.linear, // Faster, or bilinear for quality
      );

      // Prepare input for UINT8 model: [1, H, W, C]
      // tflite_flutter expects a List<dynamic> that matches the tensor structure.
      final Uint8List imagePixelBytes = resizedImage.getBytes(
        order: img_lib.ChannelOrder.rgb,
      );
      var inputTensorData = List.generate(
        1,
        (_) => List.generate(
          _tfliteInputHeight!,
          (y) => List.generate(_tfliteInputWidth!, (x) {
            int pixelIdx = (y * _tfliteInputWidth! + x) * 3;
            return [
              imagePixelBytes[pixelIdx + 0], // R
              imagePixelBytes[pixelIdx + 1], // G
              imagePixelBytes[pixelIdx + 2], // B
            ];
          }, growable: false),
          growable: false,
        ),
        growable: false,
      );

      // Prepare output buffers based on _tfliteMaxDetections
      // Shapes from TFLite SSD MobileNet: Boxes [1,N,4], Classes [1,N], Scores [1,N], NumDetections [1]
      List<List<List<double>>> outputBoxes = List.generate(
        1,
        (_) => List.generate(_tfliteMaxDetections, (_) => List.filled(4, 0.0)),
      );
      List<List<double>> outputClasses = List.generate(
        1,
        (_) => List.filled(_tfliteMaxDetections, 0.0),
      );
      List<List<double>> outputScores = List.generate(
        1,
        (_) => List.filled(_tfliteMaxDetections, 0.0),
      );
      List<double> outputNumDetections = List.filled(1, 0.0);

      Map<int, Object> outputs = {
        _tfBoxesIdx: outputBoxes,
        _tfClassesIdx: outputClasses,
        _tfScoresIdx: outputScores,
        _tfNumDetectionsIdx: outputNumDetections,
      };

      _tfliteInterpreter!.runForMultipleInputs([inputTensorData], outputs);

      final List<List<double>> boxes = outputBoxes[0];
      final List<double> classes = outputClasses[0];
      final List<double> scores = outputScores[0];
      final int numDetections = outputNumDetections[0].toInt().clamp(
        0,
        _tfliteMaxDetections,
      );

      List<DetectedObject> newDetections = [];
      const double confidenceThreshold = 0.45; // Adjust as needed

      for (int i = 0; i < numDetections; i++) {
        if (i >= scores.length || i >= classes.length || i >= boxes.length)
          break;

        if (scores[i] > confidenceThreshold) {
          // Boxes typically [ymin, xmin, ymax, xmax], normalized 0.0-1.0
          final double yMin = (boxes[i][0] * originalHeight).clamp(
            0.0,
            originalHeight.toDouble(),
          );
          final double xMin = (boxes[i][1] * originalWidth).clamp(
            0.0,
            originalWidth.toDouble(),
          );
          final double yMax = (boxes[i][2] * originalHeight).clamp(
            0.0,
            originalHeight.toDouble(),
          );
          final double xMax = (boxes[i][3] * originalWidth).clamp(
            0.0,
            originalWidth.toDouble(),
          );

          if (xMax > xMin && yMax > yMin) {
            // Valid box
            newDetections.add(
              DetectedObject(
                boundingBox: Rect.fromLTRB(xMin, yMin, xMax, yMax),
                classId: classes[i].round(), // Class ID is float, round to int
                confidence: scores[i],
              ),
            );
          }
        }
      }

      newDetections = _applyNMS(newDetections, 0.4); // Non-Maximal Suppression

      if (mounted) {
        setState(() {
          _detectedObjects = newDetections;
          _originalImageSizeForPainter = Size(
            originalWidth.toDouble(),
            originalHeight.toDouble(),
          );
        });
      }
      stopwatch.stop();
      debugPrint(
        "TFLite: Processed frame. Found ${newDetections.length} objects (NMS) in ${stopwatch.elapsedMilliseconds}ms. Raw Detections: $numDetections",
      );
    } catch (e, stackTrace) {
      debugPrint("TFLite: Detection Error: $e\nStackTrace: $stackTrace");
    } finally {
      if (mounted) _isTfliteProcessing = false;
    }
  }

  List<DetectedObject> _applyNMS(
    List<DetectedObject> detections,
    double iouThreshold,
  ) {
    if (detections.isEmpty) return [];
    detections.sort((a, b) => b.confidence.compareTo(a.confidence));
    List<DetectedObject> selected = [];
    List<bool> removed = List.filled(detections.length, false);
    for (int i = 0; i < detections.length; i++) {
      if (removed[i]) continue;
      selected.add(detections[i]);
      for (int j = i + 1; j < detections.length; j++) {
        if (removed[j]) continue;
        if (_calculateIOU(
              detections[i].boundingBox,
              detections[j].boundingBox,
            ) >
            iouThreshold) {
          removed[j] = true;
        }
      }
    }
    return selected;
  }

  double _calculateIOU(Rect boxA, Rect boxB) {
    double xA = boxA.left > boxB.left ? boxA.left : boxB.left;
    double yA = boxA.top > boxB.top ? boxA.top : boxB.top;
    double xB = boxA.right < boxB.right ? boxA.right : boxB.right;
    double yB = boxA.bottom < boxB.bottom ? boxA.bottom : boxB.bottom;
    double interArea =
        (xB - xA).clamp(0, double.infinity) *
        (yB - yA).clamp(0, double.infinity);
    double boxAArea = boxA.width * boxA.height;
    double boxBArea = boxB.width * boxB.height;
    double unionArea = boxAArea + boxBArea - interArea;
    return unionArea > 0 ? interArea / unionArea : 0.0;
  }

  void _updatePerformanceStats() {
    final now = DateTime.now();
    if (_lastPerfUpdateTime != null &&
        now.difference(_lastPerfUpdateTime!).inSeconds >= 1) {
      if (mounted) {
        setState(() {
          _perfStats =
              "Stream FPS: $_framesReceivedCounter, Detection Runs/s: $_tfliteRunsCounter";
        });
      }
      _framesReceivedCounter = 0;
      _tfliteRunsCounter = 0;
      _lastPerfUpdateTime = now;
    } else if (_lastPerfUpdateTime == null) {
      _lastPerfUpdateTime = now;
    }
  }

  Timer? _statusPollingTimer;
  void _startStatusPolling() {
    _stopStatusPolling(); // Ensure no duplicates
    _fetchServerStatus(); // Initial fetch
    _statusPollingTimer = Timer.periodic(const Duration(seconds: 5), (timer) {
      if (_isConnected)
        _fetchServerStatus();
      else
        timer.cancel(); // Stop if disconnected
    });
  }

  void _stopStatusPolling() {
    _statusPollingTimer?.cancel();
    _statusPollingTimer = null;
  }

  Future<void> _fetchServerStatus() async {
    if (!_isConnected || _statusApiUrl.isEmpty) return;
    try {
      final response = await http
          .get(Uri.parse(_statusApiUrl))
          .timeout(const Duration(seconds: 3));
      if (response.statusCode == 200 && mounted) {
        setState(() {
          _serverStatus = jsonDecode(response.body);
        });
      }
    } catch (e) {
      debugPrint("Error fetching server status: $e");
      if (mounted && _isConnected) {
        // Only show error if still supposed to be connected
        // setState(() => _currentStatusMessage = "Status poll failed.");
      }
    }
  }

  void _showSnackBar(String message) {
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text(message), duration: const Duration(seconds: 3)),
      );
    }
  }

  // Helper for finding byte patterns in a list
  int _indexOfBytes(List<int> source, List<int> pattern, [int startIndex = 0]) {
    if (pattern.isEmpty ||
        source.isEmpty ||
        startIndex < 0 ||
        startIndex > source.length - pattern.length)
      return -1;
    for (int i = startIndex; i <= source.length - pattern.length; i++) {
      bool found = true;
      for (int j = 0; j < pattern.length; j++) {
        if (source[i + j] != pattern[j]) {
          found = false;
          break;
        }
      }
      if (found) return i;
    }
    return -1;
  }

  int _lastIndexOfBytes(List<int> source, List<int> pattern) {
    if (pattern.isEmpty || source.isEmpty) return -1;
    for (int i = source.length - pattern.length; i >= 0; i--) {
      bool found = true;
      for (int j = 0; j < pattern.length; j++) {
        if (source[i + j] != pattern[j]) {
          found = false;
          break;
        }
      }
      if (found) return i;
    }
    return -1;
  }

  @override
  void dispose() {
    _ipController.dispose();
    _tfliteInterpreter?.close();
    _stopVideoStream();
    _stopStatusPolling();
    super.dispose();
  }

  // --- UI Builder Methods ---
  Widget _buildIpInputRow() => Padding(
    padding: const EdgeInsets.symmetric(vertical: 8.0),
    child: Row(
      children: <Widget>[
        Expanded(
          child: TextField(
            controller: _ipController,
            decoration: const InputDecoration(
              labelText: 'Python Server IP',
              hintText: 'e.g., 192.168.1.XXX',
              isDense: true,
            ),
            keyboardType: const TextInputType.numberWithOptions(
              signed: false,
              decimal: false,
            ),
            enabled: !_isConnected,
          ),
        ),
        const SizedBox(width: 10),
        ElevatedButton.icon(
          icon: Icon(_isConnected ? Icons.link_off : Icons.link, size: 20),
          label: Text(_isConnected ? 'Disconnect' : 'Connect'),
          style: ElevatedButton.styleFrom(
            backgroundColor: _isConnected
                ? Colors.orange[700]
                : (_isTfliteModelLoaded ? Colors.teal[600] : Colors.grey[700]),
          ),
          onPressed: (_isConnected
              ? _disconnectFromServer
              : (_isTfliteModelLoaded ? _connectToServer : null)),
        ),
      ],
    ),
  );

  Widget _buildStatusDisplay() {
    String camStatusText = _serverStatus['camera_running'] == true
        ? "ON (${_serverStatus['camera_resolution']?[0]}x${_serverStatus['camera_resolution']?[1]} @ ${_serverStatus['camera_target_fps']} FPS)"
        : "OFF";
    String arduinoStatusText =
        _serverStatus['arduino_controller_status'] ?? "Unknown";

    return Card(
      elevation: 2,
      margin: const EdgeInsets.symmetric(vertical: 8.0),
      child: Padding(
        padding: const EdgeInsets.all(12.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              _currentStatusMessage,
              style: TextStyle(
                fontSize: 13,
                color: _currentStatusMessage.toLowerCase().contains("error")
                    ? Colors.redAccent
                    : Colors.white70,
              ),
              overflow: TextOverflow.ellipsis,
            ),
            const SizedBox(height: 6),
            Text(
              _perfStats,
              style: const TextStyle(fontSize: 12, color: Colors.cyanAccent),
            ),
            const SizedBox(height: 6),
            Row(
              children: [
                Icon(
                  Icons.videocam,
                  size: 16,
                  color: _serverStatus['camera_running'] == true
                      ? Colors.greenAccent
                      : Colors.redAccent,
                ),
                const SizedBox(width: 5),
                Text(
                  "Camera: $camStatusText",
                  style: const TextStyle(fontSize: 12),
                ),
              ],
            ),
            Row(
              children: [
                Icon(
                  Icons.memory,
                  size: 16,
                  color: arduinoStatusText == 'Connected'
                      ? Colors.greenAccent
                      : Colors.redAccent,
                ),
                const SizedBox(width: 5),
                Text(
                  "Arduino Ctrl: $arduinoStatusText",
                  style: const TextStyle(fontSize: 12),
                ),
              ],
            ),
            if (_isObjectDetectionEnabled && _detectedObjects.isNotEmpty)
              Padding(
                padding: const EdgeInsets.only(top: 4.0),
                child: Text(
                  'Detections: ${_detectedObjects.length}',
                  style: const TextStyle(
                    color: Colors.amberAccent,
                    fontSize: 12,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildVideoDisplay() {
    Widget videoContent;
    if (!_isConnected && !_isVideoLoading) {
      videoContent = Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(
            Icons.no_photography_outlined,
            size: 60,
            color: Colors.grey.shade600,
          ),
          const SizedBox(height: 10),
          Text(
            'Connect to view stream',
            style: TextStyle(color: Colors.grey.shade400),
          ),
        ],
      );
    } else if (_isVideoLoading) {
      videoContent = const Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          CircularProgressIndicator(),
          SizedBox(height: 16),
          Text(
            'Connecting to video...',
            style: TextStyle(color: Colors.white70),
          ),
        ],
      );
    } else if (_currentFrameBytes == null && _isConnected) {
      videoContent = Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const Icon(Icons.videocam_off, size: 60, color: Colors.orangeAccent),
          const SizedBox(height: 10),
          const Text(
            'Waiting for video frames...',
            style: TextStyle(color: Colors.white70),
          ),
          const SizedBox(height: 10),
          ElevatedButton(
            onPressed: () async {
              setState(() {
                _isVideoLoading = true;
                _currentStatusMessage = 'Retrying video connection...';
              });
              await _startVideoStream(); // Attempt to restart
            },
            child: const Text('Retry Video'),
          ),
        ],
      );
    } else if (_currentFrameBytes != null) {
      videoContent = Stack(
        fit: StackFit.expand,
        children: [
          Image.memory(
            _currentFrameBytes!,
            fit: BoxFit.contain,
            gaplessPlayback: true, // gapless for smoother updates
            errorBuilder: (context, error, stackTrace) {
              debugPrint('Error displaying frame in Image.memory: $error');
              return const Center(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Icon(Icons.error_outline, color: Colors.red, size: 40),
                    SizedBox(height: 8),
                    Text(
                      'Frame display error',
                      style: TextStyle(color: Colors.redAccent),
                    ),
                  ],
                ),
              );
            },
          ),
          if (_isObjectDetectionEnabled &&
              _isTfliteModelLoaded &&
              _originalImageSizeForPainter != null &&
              _detectedObjects.isNotEmpty)
            Positioned.fill(
              child: CustomPaint(
                painter: ObjectBoxPainter(
                  detectedObjects: _detectedObjects,
                  originalImageSize: _originalImageSizeForPainter!,
                ),
              ),
            ),
        ],
      );
    } else {
      videoContent = Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(Icons.error_outline, size: 60, color: Colors.red.shade400),
          const SizedBox(height: 10),
          Text(
            'Video unavailable or error',
            style: TextStyle(color: Colors.red.shade300),
          ),
        ],
      );
    }

    return AspectRatio(
      aspectRatio: 4 / 3, // Common camera aspect ratio
      child: Container(
        decoration: BoxDecoration(
          color: Colors.black,
          borderRadius: BorderRadius.circular(8),
          border: Border.all(color: Colors.teal[700]!, width: 2),
        ),
        child: ClipRRect(
          borderRadius: BorderRadius.circular(6),
          child: videoContent,
        ),
      ),
    );
  }

  Widget _buildControlButton(
    String label,
    String direction,
    IconData icon, {
    bool isStop = false,
  }) {
    return Expanded(
      child: Padding(
        padding: const EdgeInsets.all(5.0),
        child: GestureDetector(
          onTapDown: (_) => _sendControlCommand(direction),
          onTapUp: (_) {
            if (!isStop) _sendControlCommand('stop');
          }, // Send stop on release for movement
          onTapCancel: () {
            if (!isStop) _sendControlCommand('stop');
          },
          onTap: () {
            if (isStop) _sendControlCommand('stop');
          }, // For stop button, tap is enough
          child: Container(
            padding: const EdgeInsets.symmetric(vertical: 18, horizontal: 12),
            decoration: BoxDecoration(
              color: isStop
                  ? Colors.red[700]
                  : Theme.of(
                      context,
                    ).colorScheme.primaryContainer.withOpacity(0.8),
              borderRadius: BorderRadius.circular(12),
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withOpacity(0.4),
                  spreadRadius: 1,
                  blurRadius: 5,
                  offset: const Offset(0, 3),
                ),
              ],
            ),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(icon, size: 28, color: Colors.white),
                if (label.isNotEmpty) const SizedBox(height: 5),
                if (label.isNotEmpty)
                  Text(
                    label,
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 11,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildControls() => Padding(
    padding: const EdgeInsets.only(top: 16.0),
    child: Column(
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
          children: [
            const Spacer(),
            _buildControlButton('Forward', 'up', Icons.arrow_upward_rounded),
            const Spacer(),
          ],
        ),
        const SizedBox(height: 8),
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
          children: [
            _buildControlButton('Left', 'left', Icons.arrow_back_rounded),
            _buildControlButton(
              'STOP',
              'stop',
              Icons.stop_circle_outlined,
              isStop: true,
            ),
            _buildControlButton('Right', 'right', Icons.arrow_forward_rounded),
          ],
        ),
        const SizedBox(height: 8),
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
          children: [
            const Spacer(),
            _buildControlButton(
              'Backward',
              'down',
              Icons.arrow_downward_rounded,
            ),
            const Spacer(),
          ],
        ),
      ],
    ),
  );

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Car Remote Control'),
        actions: [
          IconButton(
            icon: Icon(
              _isObjectDetectionEnabled
                  ? Icons.visibility_rounded
                  : Icons.visibility_off_rounded,
              color: _isObjectDetectionEnabled && _isTfliteModelLoaded
                  ? Colors.lightGreenAccent[400]
                  : (_isTfliteModelLoaded ? null : Colors.grey[600]),
            ),
            onPressed: _isTfliteModelLoaded ? _toggleObjectDetection : null,
            tooltip: _isTfliteModelLoaded
                ? 'Toggle TFLite Detection'
                : 'TFLite Model Not Loaded',
          ),
        ],
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          // Allows scrolling if content overflows on small screens
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: <Widget>[
              _buildIpInputRow(),
              _buildStatusDisplay(),
              const SizedBox(height: 10),
              _buildVideoDisplay(),
              _buildControls(),
            ],
          ),
        ),
      ),
    );
  }
}

// Custom Painter for drawing bounding boxes
class ObjectBoxPainter extends CustomPainter {
  final List<DetectedObject> detectedObjects;
  final Size originalImageSize;

  ObjectBoxPainter({
    required this.detectedObjects,
    required this.originalImageSize,
  });

  @override
  void paint(Canvas canvas, Size displaySize) {
    // displaySize is the size of the widget displaying the image
    if (originalImageSize.isEmpty ||
        displaySize.isEmpty ||
        detectedObjects.isEmpty)
      return;

    // Calculate scaling factors to fit the original image size into the displaySize (like BoxFit.contain)
    final double scaleX = displaySize.width / originalImageSize.width;
    final double scaleY = displaySize.height / originalImageSize.height;
    final double scale = scaleX < scaleY ? scaleX : scaleY;

    // Calculate offsets to center the scaled image within the displaySize
    final double offsetX =
        (displaySize.width - originalImageSize.width * scale) / 2;
    final double offsetY =
        (displaySize.height - originalImageSize.height * scale) / 2;

    final boxPaint = Paint()
      ..strokeWidth = 2.0
      ..style = PaintingStyle.stroke;

    for (var obj in detectedObjects) {
      // Determine color based on class or confidence (example)
      boxPaint.color = obj.className == 'person'
          ? Colors.redAccent.withOpacity(0.85)
          : (obj.className.startsWith('ClassID')
                ? Colors.orangeAccent.withOpacity(0.85)
                : Colors.greenAccent.withOpacity(0.85));

      // Scale and offset the bounding box coordinates
      final Rect scaledRect = Rect.fromLTRB(
        obj.boundingBox.left * scale + offsetX,
        obj.boundingBox.top * scale + offsetY,
        obj.boundingBox.right * scale + offsetX,
        obj.boundingBox.bottom * scale + offsetY,
      );
      canvas.drawRect(scaledRect, boxPaint);

      final String labelText =
          '${obj.className} (${(obj.confidence * 100).toStringAsFixed(0)}%)';
      TextPainter textPainter = TextPainter(
        text: TextSpan(
          text: ' $labelText ', // Padding for text background
          style: TextStyle(
            color: boxPaint.color.computeLuminance() > 0.5
                ? Colors.black87
                : Colors.white,
            fontSize: 11,
            fontWeight: FontWeight.w500,
            backgroundColor: boxPaint.color.withOpacity(0.7),
            shadows: [
              BoxShadow(
                color: Colors.black.withOpacity(0.3),
                blurRadius: 1,
                offset: const Offset(0, 1),
              ),
            ],
          ),
        ),
        textDirection: TextDirection.ltr,
      );
      textPainter.layout(minWidth: 0, maxWidth: displaySize.width);

      // Position the label above the box, ensuring it's within bounds
      double labelX = scaledRect.left;
      double labelY = scaledRect.top - textPainter.height - 3;
      if (labelY < 0)
        labelY = scaledRect.bottom + 3; // If not enough space above, put below
      if (labelY + textPainter.height > displaySize.height)
        labelY =
            scaledRect.top -
            textPainter.height -
            3; // Re-check if below goes out
      if (labelY < 0) labelY = 0; // Final clamp

      if (labelX < 0) labelX = 0;
      if (labelX + textPainter.width > displaySize.width)
        labelX = displaySize.width - textPainter.width;

      textPainter.paint(canvas, Offset(labelX, labelY));
    }
  }

  @override
  bool shouldRepaint(covariant ObjectBoxPainter oldDelegate) {
    // Repaint if objects or image size changes, or if list content might have changed
    return oldDelegate.detectedObjects != detectedObjects ||
        oldDelegate.originalImageSize != originalImageSize ||
        oldDelegate.detectedObjects.length != detectedObjects.length;
  }
}
