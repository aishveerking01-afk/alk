import 'dart:async';
import 'dart:io';

import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:image_picker/image_picker.dart';
import 'package:llamafu/llamafu.dart';
import 'package:lucide_icons/lucide_icons.dart';
import 'package:path_provider/path_provider.dart';
import 'package:speech_to_text/speech_to_text.dart' as stt;

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const RuiApp());
}

// ---------------------------------------------------------------------------
// Theme
// ---------------------------------------------------------------------------

class RuiColors {
  static const background = Color(0xFF121212);
  static const surface = Color(0xFF1C1C1E);
  static const surfaceHigh = Color(0xFF2A2A2D);
  static const userBubble = Color(0xFF2F3136);
  static const accent = Color(0xFF10A37F);
  static const text = Color(0xFFECECEC);
  static const textDim = Color(0xFF9A9AA0);
  static const danger = Color(0xFFE5534B);
  static const warning = Color(0xFFE3B341);
}

class RuiApp extends StatelessWidget {
  const RuiApp({super.key});

  @override
  Widget build(BuildContext context) {
    final base = ThemeData.dark(useMaterial3: true);
    return MaterialApp(
      title: 'Rui AI',
      debugShowCheckedModeBanner: false,
      theme: base.copyWith(
        scaffoldBackgroundColor: RuiColors.background,
        colorScheme: const ColorScheme.dark(
          primary: RuiColors.accent,
          surface: RuiColors.surface,
        ),
        textTheme: GoogleFonts.interTextTheme(base.textTheme)
            .apply(bodyColor: RuiColors.text, displayColor: RuiColors.text),
        appBarTheme: AppBarTheme(
          backgroundColor: RuiColors.background,
          elevation: 0,
          scrolledUnderElevation: 0,
          titleTextStyle: GoogleFonts.inter(
              fontSize: 18, fontWeight: FontWeight.w600, color: RuiColors.text),
        ),
        drawerTheme:
            const DrawerThemeData(backgroundColor: RuiColors.background),
      ),
      home: const ChatScreen(),
    );
  }
}

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------

class ModelFileSpec {
  final String url;
  final String fileName;
  const ModelFileSpec(this.url, this.fileName);
}

class ModelSpec {
  final String id;
  final String name;
  final String subtitle;
  final String sizeLabel;
  final bool isVision;
  final List<ModelFileSpec> files; // files.first = main model, files[1] = mmproj
  const ModelSpec({
    required this.id,
    required this.name,
    required this.subtitle,
    required this.sizeLabel,
    required this.isVision,
    required this.files,
  });
}

// Verify the Bunny filenames against the repo's "Files" tab before shipping.
const List<ModelSpec> kModelCatalog = [
  ModelSpec(
    id: 'gemma2b',
    name: 'Gemma 2B (Fast Text)',
    subtitle: 'Gemma 2 2B Instruct, Q4_K_M',
    sizeLabel: '~1.7 GB',
    isVision: false,
    files: [
      ModelFileSpec(
        'https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf',
        'gemma-2-2b-it-Q4_K_M.gguf',
      ),
    ],
  ),
  ModelSpec(
    id: 'bunny3b',
    name: 'Bunny 3B (Vision/Image-to-Text)',
    subtitle: 'Bunny v1.0 3B + SigLIP projector',
    sizeLabel: '~2.5 GB',
    isVision: true,
    files: [
      ModelFileSpec(
        'https://huggingface.co/BAAI/Bunny-v1_0-3B-gguf/resolve/main/ggml-model-Q4_K_M.gguf',
        'bunny-3b-Q4_K_M.gguf',
      ),
      ModelFileSpec(
        'https://huggingface.co/BAAI/Bunny-v1_0-3B-gguf/resolve/main/mmproj-model-f16.gguf',
        'bunny-3b-mmproj-f16.gguf',
      ),
    ],
  ),
];

class ChatMessage {
  final bool isUser;
  String text;
  final String? imagePath;
  bool isStreaming;
  ChatMessage({
    required this.isUser,
    required this.text,
    this.imagePath,
    this.isStreaming = false,
  });
}

enum EngineStatus { offline, loading, ready, error }

// ---------------------------------------------------------------------------
// Engine wrapper: every llamafu call lives here, so API differences between
// llamafu versions only need to be fixed in this one class.
// ---------------------------------------------------------------------------

class RuiEngine {
  Llamafu? _llamafu;
  ModelSpec? spec;

  bool get isLoaded => _llamafu != null;

  Future<void> load(ModelSpec model, List<String> localPaths) async {
    await unload();
    _llamafu = await Llamafu.init(
      modelPath: localPaths.first,
      mmprojPath: model.isVision && localPaths.length > 1 ? localPaths[1] : null,
      threads: 4,
      contextSize: 2048,
    );
    spec = model;
  }

  Future<void> unload() async {
    _llamafu?.close();
    _llamafu = null;
    spec = null;
  }

  String _buildPrompt(List<ChatMessage> history, bool withImage) {
    final model = spec!;
    if (model.isVision) {
      final last = history.lastWhere((m) => m.isUser);
      final imageTag = withImage ? '<image>\n' : '';
      return 'A chat between a curious user and an artificial intelligence '
          'assistant. The assistant gives helpful, detailed, and polite '
          'answers to the user\'s questions. USER: $imageTag${last.text} '
          'ASSISTANT:';
    }
    // Gemma chat template, keeping the last 6 turns to respect the context.
    final turns = history.where((m) => m.text.trim().isNotEmpty).toList();
    final recent =
        turns.length > 6 ? turns.sublist(turns.length - 6) : turns;
    final buf = StringBuffer();
    for (final m in recent) {
      buf.write(m.isUser
          ? '<start_of_turn>user\n${m.text.trim()}<end_of_turn>\n'
          : '<start_of_turn>model\n${m.text.trim()}<end_of_turn>\n');
    }
    buf.write('<start_of_turn>model\n');
    return buf.toString();
  }

  /// Streams tokens. If an image path is given, it is passed straight to the
  /// llama runtime through the multimodal call.
  Stream<String> generate(List<ChatMessage> history, {File? image}) async* {
    final engine = _llamafu;
    if (engine == null || spec == null) {
      throw StateError('No model is loaded.');
    }
    final hasImage = image != null && spec!.isVision;
    final prompt = _buildPrompt(history, hasImage);

    if (hasImage) {
      // Multimodal completion returns the full text; reveal it progressively
      // so the UI still streams like a chat assistant.
      final result = await engine.multimodalComplete(
        prompt,
        mediaInputs: [
          MediaInput(type: MediaType.image, data: image.path),
        ],
        maxTokens: 512,
        temperature: 0.4,
      );
      final words = result.split(RegExp(r'(?<=\s)'));
      for (final w in words) {
        yield w;
        await Future<void>.delayed(const Duration(milliseconds: 18));
      }
      return;
    }

    await for (final token in engine.completeStream(
      prompt: prompt,
      maxTokens: 512,
      temperature: 0.7,
    )) {
      yield token;
    }
  }
}

// ---------------------------------------------------------------------------
// Chat screen
// ---------------------------------------------------------------------------

class ChatScreen extends StatefulWidget {
  const ChatScreen({super.key});

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final GlobalKey<ScaffoldState> _scaffoldKey = GlobalKey<ScaffoldState>();
  final TextEditingController _controller = TextEditingController();
  final ScrollController _scroll = ScrollController();
  final ImagePicker _picker = ImagePicker();
  final stt.SpeechToText _speech = stt.SpeechToText();
  final Dio _dio = Dio();
  final RuiEngine _engine = RuiEngine();

  final List<ChatMessage> _messages = [];
  final Map<String, double> _progress = {}; // 0.0 - 1.0
  final Set<String> _downloading = {};
  final Set<String> _downloaded = {};
  final Map<String, CancelToken> _cancelTokens = {};

  File? _attachedImage;
  EngineStatus _status = EngineStatus.offline;
  String? _loadedModelId;
  bool _generating = false;
  bool _stopRequested = false;
  bool _speechReady = false;
  bool _listening = false;
  String _dictationPrefix = '';
  String _statusDetail = 'No model loaded';

  @override
  void initState() {
    super.initState();
    _refreshDownloadedModels();
  }

  @override
  void dispose() {
    for (final t in _cancelTokens.values) {
      t.cancel();
    }
    _speech.cancel();
    _engine.unload();
    _controller.dispose();
    _scroll.dispose();
    super.dispose();
  }

  // ------------------------------- Storage ---------------------------------

  Future<Directory> _modelDir() async {
    final base = await getApplicationDocumentsDirectory();
    final dir = Directory('${base.path}/models');
    if (!await dir.exists()) await dir.create(recursive: true);
    return dir;
  }

  Future<List<String>> _localPaths(ModelSpec m) async {
    final dir = await _modelDir();
    return m.files.map((f) => '${dir.path}/${f.fileName}').toList();
  }

  Future<void> _refreshDownloadedModels() async {
    for (final m in kModelCatalog) {
      final paths = await _localPaths(m);
      final complete = paths.every((p) => File(p).existsSync());
      if (complete) {
        _downloaded.add(m.id);
      } else {
        _downloaded.remove(m.id);
      }
    }
    if (mounted) setState(() {});
  }

  // ------------------------------ Downloads --------------------------------

  Future<void> _downloadModel(ModelSpec m, void Function() refreshSheet) async {
    if (_downloading.contains(m.id)) return;
    final token = CancelToken();
    _cancelTokens[m.id] = token;
    _downloading.add(m.id);
    _progress[m.id] = 0.0;
    setState(() {});
    refreshSheet();

    try {
      final paths = await _localPaths(m);
      for (var i = 0; i < m.files.length; i++) {
        final finalPath = paths[i];
        if (File(finalPath).existsSync()) {
          _progress[m.id] = (i + 1) / m.files.length;
          setState(() {});
          refreshSheet();
          continue;
        }
        final partPath = '$finalPath.part';
        await _dio.download(
          m.files[i].url,
          partPath,
          cancelToken: token,
          deleteOnError: true,
          options: Options(
            followRedirects: true,
            receiveTimeout: const Duration(minutes: 30),
          ),
          onReceiveProgress: (received, total) {
            if (total <= 0) return;
            final fileFraction = received / total;
            _progress[m.id] = (i + fileFraction) / m.files.length;
            if (mounted) {
              setState(() {});
              refreshSheet();
            }
          },
        );
        await File(partPath).rename(finalPath);
      }
      _downloaded.add(m.id);
      _progress[m.id] = 1.0;
      _showSnack('${m.name} downloaded. Tap "Load model" to start.');
    } on DioException catch (e) {
      if (CancelToken.isCancel(e)) {
        _progress.remove(m.id);
        _showSnack('Download cancelled.');
      } else {
        _progress.remove(m.id);
        _showSnack(
            'Download failed: ${e.message ?? 'network error'}. Check your connection and the model URL.');
      }
    } catch (e) {
      _progress.remove(m.id);
      _showSnack('Download failed: $e');
    } finally {
      _downloading.remove(m.id);
      _cancelTokens.remove(m.id);
      if (mounted) {
        setState(() {});
        refreshSheet();
      }
    }
  }

  void _cancelDownload(ModelSpec m) {
    _cancelTokens[m.id]?.cancel();
  }

  Future<void> _deleteModel(ModelSpec m, void Function() refreshSheet) async {
    if (_loadedModelId == m.id) {
      await _engine.unload();
      _loadedModelId = null;
      _status = EngineStatus.offline;
      _statusDetail = 'No model loaded';
    }
    for (final p in await _localPaths(m)) {
      final f = File(p);
      if (await f.exists()) await f.delete();
    }
    _progress.remove(m.id);
    await _refreshDownloadedModels();
    refreshSheet();
  }

  // ------------------------------ Engine -----------------------------------

  Future<void> _loadModel(ModelSpec m, void Function() refreshSheet) async {
    setState(() {
      _status = EngineStatus.loading;
      _statusDetail = 'Loading ${m.name}...';
    });
    refreshSheet();
    try {
      final paths = await _localPaths(m);
      await _engine.load(m, paths);
      if (!mounted) return;
      setState(() {
        _loadedModelId = m.id;
        _status = EngineStatus.ready;
        _statusDetail = '${m.name} ready';
        if (!m.isVision) _attachedImage = null;
      });
      _showSnack('${m.name} is ready. Everything runs on this device.');
      if (mounted && Navigator.of(context).canPop()) {
        Navigator.of(context).pop();
      }
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _status = EngineStatus.error;
        _statusDetail = 'Failed to load model';
      });
      _showSnack(
          'Could not load the model: $e. Close other apps to free RAM and try again.');
    }
    refreshSheet();
  }

  // ------------------------------ Sending ----------------------------------

  Future<void> _send() async {
    final text = _controller.text.trim();
    if (text.isEmpty || _generating) return;

    if (!_engine.isLoaded) {
      _showSnack('Load a model first from the model manager.');
      _openModelManager();
      return;
    }

    if (_listening) await _stopListening();

    final image = _attachedImage;
    final modelIsVision = _engine.spec?.isVision ?? false;
    if (image != null && !modelIsVision) {
      _showSnack(
          'Images need the Bunny 3B vision model. Load it or remove the image.');
      return;
    }

    final userMsg = ChatMessage(
      isUser: true,
      text: text,
      imagePath: image?.path,
    );
    final replyMsg = ChatMessage(isUser: false, text: '', isStreaming: true);

    setState(() {
      _messages.add(userMsg);
      _messages.add(replyMsg);
      _controller.clear();
      _attachedImage = null;
      _generating = true;
      _stopRequested = false;
    });
    _scrollToBottom();

    try {
      final history =
          _messages.where((m) => m != replyMsg).toList(growable: false);
      await for (final token in _engine.generate(history, image: image)) {
        if (_stopRequested || !mounted) break;
        setState(() => replyMsg.text += token);
        _scrollToBottom();
      }
      if (replyMsg.text.trim().isEmpty && !_stopRequested) {
        setState(() => replyMsg.text = '(No response was generated.)');
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          replyMsg.text = replyMsg.text.isEmpty
              ? 'Something went wrong while generating: $e'
              : '${replyMsg.text}\n\n[Generation interrupted: $e]';
        });
      }
    } finally {
      if (mounted) {
        setState(() {
          replyMsg.isStreaming = false;
          _generating = false;
          _stopRequested = false;
        });
        _scrollToBottom();
      }
    }
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!_scroll.hasClients) return;
      _scroll.jumpTo(_scroll.position.maxScrollExtent);
    });
  }

  // ------------------------------ Images -----------------------------------

  Future<void> _pickImage(ImageSource source) async {
    try {
      final picked =
          await _picker.pickImage(source: source, maxWidth: 1024, imageQuality: 85);
      if (picked == null) return;
      setState(() => _attachedImage = File(picked.path));
      if (!(_engine.spec?.isVision ?? false)) {
        _showSnack('Load Bunny 3B (Vision) to ask questions about this image.');
      }
    } catch (e) {
      _showSnack('Could not access the image: $e. Check app permissions.');
    }
  }

  void _openAttachmentSheet() {
    showModalBottomSheet(
      context: context,
      backgroundColor: RuiColors.surface,
      shape: const RoundedRectangleBorder(
          borderRadius: BorderRadius.vertical(top: Radius.circular(20))),
      builder: (ctx) => SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(vertical: 12),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              ListTile(
                leading: const Icon(LucideIcons.image),
                title: const Text('Choose from gallery'),
                onTap: () {
                  Navigator.pop(ctx);
                  _pickImage(ImageSource.gallery);
                },
              ),
              ListTile(
                leading: const Icon(LucideIcons.camera),
                title: const Text('Take a photo'),
                onTap: () {
                  Navigator.pop(ctx);
                  _pickImage(ImageSource.camera);
                },
              ),
            ],
          ),
        ),
      ),
    );
  }

  // ------------------------------- Voice -----------------------------------

  Future<bool> _ensureSpeech() async {
    if (_speechReady) return true;
    try {
      _speechReady = await _speech.initialize(
        onError: (e) {
          if (!mounted) return;
          setState(() => _listening = false);
          if (e.errorMsg == 'error_permission') {
            _showSnack(
                'Microphone permission denied. Enable it in Android Settings > Apps > Rui AI > Permissions.');
          } else if (e.errorMsg != 'error_no_match' &&
              e.errorMsg != 'error_speech_timeout') {
            _showSnack('Voice error: ${e.errorMsg}');
          }
        },
        onStatus: (s) {
          if (!mounted) return;
          if (s == 'done' || s == 'notListening') {
            setState(() => _listening = false);
          }
        },
      );
    } catch (e) {
      _speechReady = false;
    }
    if (!_speechReady) {
      _showSnack(
          'Voice input is unavailable. Grant microphone permission and make sure an offline speech pack is installed.');
    }
    return _speechReady;
  }

  Future<void> _toggleListening() async {
    if (_listening) {
      await _stopListening();
      return;
    }
    if (!await _ensureSpeech()) return;

    _dictationPrefix = _controller.text.isEmpty ? '' : '${_controller.text} ';
    setState(() => _listening = true);
    try {
      await _speech.listen(
        onResult: (result) {
          if (!mounted) return;
          final text = '$_dictationPrefix${result.recognizedWords}';
          _controller.value = TextEditingValue(
            text: text,
            selection: TextSelection.collapsed(offset: text.length),
          );
        },
        listenFor: const Duration(minutes: 2),
        pauseFor: const Duration(seconds: 6),
        listenOptions: stt.SpeechListenOptions(
          onDevice: true,
          partialResults: true,
          cancelOnError: true,
          listenMode: stt.ListenMode.dictation,
        ),
      );
    } catch (e) {
      setState(() => _listening = false);
      _showSnack('Could not start listening: $e');
    }
  }

  Future<void> _stopListening() async {
    await _speech.stop();
    if (mounted) setState(() => _listening = false);
  }

  // ------------------------------- UI --------------------------------------

  void _showSnack(String msg) {
    if (!mounted) return;
    ScaffoldMessenger.of(context)
      ..hideCurrentSnackBar()
      ..showSnackBar(SnackBar(
        content: Text(msg),
        backgroundColor: RuiColors.surfaceHigh,
        behavior: SnackBarBehavior.floating,
      ));
  }

  void _openModelManager() {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: RuiColors.surface,
      shape: const RoundedRectangleBorder(
          borderRadius: BorderRadius.vertical(top: Radius.circular(20))),
      builder: (ctx) => StatefulBuilder(
        builder: (ctx, setSheetState) {
          void refresh() {
            if (ctx.mounted) setSheetState(() {});
          }

          return DraggableScrollableSheet(
            expand: false,
            initialChildSize: 0.72,
            minChildSize: 0.4,
            maxChildSize: 0.95,
            builder: (ctx, scrollController) => ListView(
              controller: scrollController,
              padding: const EdgeInsets.fromLTRB(16, 12, 16, 24),
              children: [
                Center(
                  child: Container(
                    width: 40,
                    height: 4,
                    decoration: BoxDecoration(
                        color: RuiColors.textDim.withOpacity(0.4),
                        borderRadius: BorderRadius.circular(2)),
                  ),
                ),
                const SizedBox(height: 16),
                Text('Model Manager',
                    style: GoogleFonts.inter(
                        fontSize: 20, fontWeight: FontWeight.w700)),
                const SizedBox(height: 4),
                const Text(
                  'Download once over the internet. After that, Rui runs fully offline on this device.',
                  style: TextStyle(color: RuiColors.textDim, fontSize: 13),
                ),
                const SizedBox(height: 16),
                for (final m in kModelCatalog) _modelCard(m, refresh),
              ],
            ),
          );
        },
      ),
    );
  }

  Widget _modelCard(ModelSpec m, void Function() refresh) {
    final isDownloading = _downloading.contains(m.id);
    final isDownloaded = _downloaded.contains(m.id);
    final isLoaded = _loadedModelId == m.id && _engine.isLoaded;
    final progress = _progress[m.id] ?? 0.0;

    Widget action;
    if (isDownloading) {
      action = OutlinedButton.icon(
        onPressed: () => _cancelDownload(m),
        icon: const Icon(LucideIcons.x, size: 16),
        label: const Text('Cancel'),
      );
    } else if (isLoaded) {
      action = const Chip(
        avatar: Icon(LucideIcons.check, size: 16, color: RuiColors.accent),
        label: Text('Loaded'),
        backgroundColor: RuiColors.surfaceHigh,
      );
    } else if (isDownloaded) {
      action = FilledButton.icon(
        style: FilledButton.styleFrom(backgroundColor: RuiColors.accent),
        onPressed:
            _status == EngineStatus.loading ? null : () => _loadModel(m, refresh),
        icon: const Icon(LucideIcons.cpu, size: 16),
        label: const Text('Load model'),
      );
    } else {
      action = FilledButton.icon(
        style: FilledButton.styleFrom(backgroundColor: RuiColors.accent),
        onPressed: () => _downloadModel(m, refresh),
        icon: const Icon(LucideIcons.download, size: 16),
        label: Text('Download ${m.sizeLabel}'),
      );
    }

    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: RuiColors.surfaceHigh,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(
            color: isLoaded ? RuiColors.accent : Colors.transparent),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(m.isVision ? LucideIcons.image : LucideIcons.sparkles,
                  color: RuiColors.accent, size: 20),
              const SizedBox(width: 10),
              Expanded(
                child: Text(m.name,
                    style: GoogleFonts.inter(
                        fontWeight: FontWeight.w600, fontSize: 15)),
              ),
              if (isDownloaded && !isDownloading)
                IconButton(
                  tooltip: 'Delete from device',
                  icon: const Icon(LucideIcons.trash2,
                      size: 18, color: RuiColors.textDim),
                  onPressed: () => _deleteModel(m, refresh),
                ),
            ],
          ),
          const SizedBox(height: 4),
          Text(m.subtitle,
              style: const TextStyle(color: RuiColors.textDim, fontSize: 12)),
          if (isDownloading) ...[
            const SizedBox(height: 12),
            ClipRRect(
              borderRadius: BorderRadius.circular(4),
              child: LinearProgressIndicator(
                value: progress,
                minHeight: 6,
                backgroundColor: RuiColors.background,
                color: RuiColors.accent,
              ),
            ),
            const SizedBox(height: 6),
            Text('${(progress * 100).toStringAsFixed(1)}%',
                style:
                    const TextStyle(color: RuiColors.textDim, fontSize: 12)),
          ],
          const SizedBox(height: 12),
          Align(alignment: Alignment.centerRight, child: action),
        ],
      ),
    );
  }

  Color get _statusColor {
    switch (_status) {
      case EngineStatus.ready:
        return RuiColors.accent;
      case EngineStatus.loading:
        return RuiColors.warning;
      case EngineStatus.error:
        return RuiColors.danger;
      case EngineStatus.offline:
        return RuiColors.textDim;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      key: _scaffoldKey,
      appBar: AppBar(
        leading: IconButton(
          icon: const Icon(LucideIcons.menu),
          onPressed: _openModelManager,
          tooltip: 'Model manager',
        ),
        title: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Text('Rui AI'),
            const SizedBox(width: 10),
            Tooltip(
              message: _statusDetail,
              child: Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                decoration: BoxDecoration(
                  color: _statusColor.withOpacity(0.15),
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Container(
                      width: 8,
                      height: 8,
                      decoration: BoxDecoration(
                          color: _statusColor, shape: BoxShape.circle),
                    ),
                    const SizedBox(width: 6),
                    Text(
                      switch (_status) {
                        EngineStatus.ready => 'On-device',
                        EngineStatus.loading => 'Loading',
                        EngineStatus.error => 'Error',
                        EngineStatus.offline => 'No model',
                      },
                      style: TextStyle(
                          fontSize: 11,
                          color: _statusColor,
                          fontWeight: FontWeight.w600),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
        actions: [
          IconButton(
            icon: const Icon(LucideIcons.cpu),
            tooltip: 'Models',
            onPressed: _openModelManager,
          ),
        ],
      ),
      body: Column(
        children: [
          Expanded(
            child: _messages.isEmpty
                ? _emptyState()
                : ListView.builder(
                    controller: _scroll,
                    padding: const EdgeInsets.fromLTRB(12, 8, 12, 12),
                    itemCount: _messages.length,
                    itemBuilder: (ctx, i) => _MessageBubble(
                        message: _messages[i], key: ObjectKey(_messages[i])),
                  ),
          ),
          _commandCenter(),
        ],
      ),
    );
  }

  Widget _emptyState() {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(32),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              width: 64,
              height: 64,
              decoration: BoxDecoration(
                color: RuiColors.accent.withOpacity(0.15),
                shape: BoxShape.circle,
              ),
              child: const Icon(LucideIcons.sparkles,
                  color: RuiColors.accent, size: 30),
            ),
            const SizedBox(height: 16),
            Text('How can I help you today?',
                style: GoogleFonts.inter(
                    fontSize: 20, fontWeight: FontWeight.w600)),
            const SizedBox(height: 8),
            Text(
              _engine.isLoaded
                  ? 'Private and offline. Nothing leaves your phone.'
                  : 'Open the model manager to download and load a model.',
              textAlign: TextAlign.center,
              style: const TextStyle(color: RuiColors.textDim),
            ),
            if (!_engine.isLoaded) ...[
              const SizedBox(height: 16),
              FilledButton.icon(
                style:
                    FilledButton.styleFrom(backgroundColor: RuiColors.accent),
                onPressed: _openModelManager,
                icon: const Icon(LucideIcons.download, size: 16),
                label: const Text('Open model manager'),
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _commandCenter() {
    return SafeArea(
      top: false,
      child: Container(
        padding: const EdgeInsets.fromLTRB(10, 8, 10, 10),
        decoration: const BoxDecoration(
          color: RuiColors.background,
          border: Border(top: BorderSide(color: Color(0xFF222225))),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            if (_attachedImage != null)
              Align(
                alignment: Alignment.centerLeft,
                child: Padding(
                  padding: const EdgeInsets.only(bottom: 8, left: 4),
                  child: Stack(
                    clipBehavior: Clip.none,
                    children: [
                      ClipRRect(
                        borderRadius: BorderRadius.circular(10),
                        child: Image.file(_attachedImage!,
                            width: 72, height: 72, fit: BoxFit.cover),
                      ),
                      Positioned(
                        top: -6,
                        right: -6,
                        child: GestureDetector(
                          onTap: () => setState(() => _attachedImage = null),
                          child: Container(
                            padding: const EdgeInsets.all(3),
                            decoration: const BoxDecoration(
                                color: RuiColors.surfaceHigh,
                                shape: BoxShape.circle),
                            child: const Icon(LucideIcons.x, size: 14),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            if (_listening)
              Padding(
                padding: const EdgeInsets.only(bottom: 6),
                child: Row(
                  children: const [
                    Icon(LucideIcons.mic, size: 14, color: RuiColors.danger),
                    SizedBox(width: 6),
                    Text('Listening... speak now',
                        style: TextStyle(
                            color: RuiColors.danger, fontSize: 12)),
                  ],
                ),
              ),
            Row(
              crossAxisAlignment: CrossAxisAlignment.end,
              children: [
                IconButton(
                  tooltip: 'Attach image',
                  icon: const Icon(LucideIcons.paperclip),
                  onPressed: _generating ? null : _openAttachmentSheet,
                ),
                Expanded(
                  child: Container(
                    padding: const EdgeInsets.symmetric(horizontal: 14),
                    decoration: BoxDecoration(
                      color: RuiColors.surface,
                      borderRadius: BorderRadius.circular(24),
                      border: Border.all(
                          color: _listening
                              ? RuiColors.danger
                              : const Color(0xFF2E2E32)),
                    ),
                    child: TextField(
                      controller: _controller,
                      minLines: 1,
                      maxLines: 6,
                      textInputAction: TextInputAction.newline,
                      decoration: InputDecoration(
                        hintText: _listening ? 'Listening...' : 'Message Rui',
                        hintStyle: const TextStyle(color: RuiColors.textDim),
                        border: InputBorder.none,
                      ),
                    ),
                  ),
                ),
                IconButton(
                  tooltip: _listening ? 'Stop listening' : 'Dictate',
                  onPressed: _generating ? null : _toggleListening,
                  icon: Icon(
                    _listening ? LucideIcons.micOff : LucideIcons.mic,
                    color: _listening ? RuiColors.danger : RuiColors.text,
                  ),
                ),
                const SizedBox(width: 2),
                _generating
                    ? IconButton.filled(
                        style: IconButton.styleFrom(
                            backgroundColor: RuiColors.text),
                        onPressed: () => setState(() => _stopRequested = true),
                        icon: const Icon(LucideIcons.square,
                            size: 18, color: RuiColors.background),
                      )
                    : IconButton.filled(
                        style: IconButton.styleFrom(
                            backgroundColor: RuiColors.accent),
                        onPressed: _send,
                        icon: const Icon(LucideIcons.arrowUp, size: 20),
                      ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Message bubble with lightweight markdown rendering
// ---------------------------------------------------------------------------

class _MessageBubble extends StatelessWidget {
  final ChatMessage message;
  const _MessageBubble({super.key, required this.message});

  @override
  Widget build(BuildContext context) {
    final isUser = message.isUser;
    final maxWidth = MediaQuery.of(context).size.width * 0.86;

    final bubble = Container(
      constraints: BoxConstraints(maxWidth: maxWidth),
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
      decoration: BoxDecoration(
        color: isUser ? RuiColors.userBubble : Colors.transparent,
        borderRadius: BorderRadius.circular(18),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          if (message.imagePath != null)
            Padding(
              padding: const EdgeInsets.only(bottom: 8),
              child: ClipRRect(
                borderRadius: BorderRadius.circular(12),
                child: Image.file(
                  File(message.imagePath!),
                  width: 200,
                  height: 200,
                  fit: BoxFit.cover,
                  errorBuilder: (_, __, ___) => Container(
                    width: 200,
                    height: 120,
                    color: RuiColors.surfaceHigh,
                    alignment: Alignment.center,
                    child: const Text('Image unavailable'),
                  ),
                ),
              ),
            ),
          if (message.text.isEmpty && message.isStreaming)
            const _TypingDots()
          else
            _MarkdownText(
                text: message.text + (message.isStreaming ? ' \u258D' : '')),
        ],
      ),
    );

    if (isUser) {
      return Padding(
        padding: const EdgeInsets.symmetric(vertical: 6),
        child: Align(alignment: Alignment.centerRight, child: bubble),
      );
    }

    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            width: 28,
            height: 28,
            margin: const EdgeInsets.only(top: 8, right: 4),
            decoration: const BoxDecoration(
                color: RuiColors.accent, shape: BoxShape.circle),
            child: const Icon(LucideIcons.sparkles, size: 15),
          ),
          Flexible(child: bubble),
        ],
      ),
    );
  }
}

class _TypingDots extends StatefulWidget {
  const _TypingDots();
  @override
  State<_TypingDots> createState() => _TypingDotsState();
}

class _TypingDotsState extends State<_TypingDots>
    with SingleTickerProviderStateMixin {
  late final AnimationController _c = AnimationController(
      vsync: this, duration: const Duration(milliseconds: 1000))
    ..repeat();

  @override
  void dispose() {
    _c.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _c,
      builder: (_, __) => Row(
        mainAxisSize: MainAxisSize.min,
        children: List.generate(3, (i) {
          final t = ((_c.value * 3) - i).clamp(0.0, 1.0);
          final o = 0.3 + 0.7 * (1 - (t - 0.5).abs() * 2).clamp(0.0, 1.0);
          return Container(
            width: 7,
            height: 7,
            margin: const EdgeInsets.symmetric(horizontal: 2, vertical: 8),
            decoration: BoxDecoration(
                color: RuiColors.textDim.withOpacity(o),
                shape: BoxShape.circle),
          );
        }),
      ),
    );
  }
}

class _MarkdownText extends StatelessWidget {
  final String text;
  const _MarkdownText({required this.text});

  static final RegExp _inline = RegExp(r'(\*\*[^*]+\*\*|`[^`]+`)');

  List<InlineSpan> _spans(String line, TextStyle base) {
    final spans = <InlineSpan>[];
    var cursor = 0;
    for (final m in _inline.allMatches(line)) {
      if (m.start > cursor) {
        spans.add(TextSpan(text: line.substring(cursor, m.start), style: base));
      }
      final token = m.group(0)!;
      if (token.startsWith('**')) {
        spans.add(TextSpan(
            text: token.substring(2, token.length - 2),
            style: base.copyWith(fontWeight: FontWeight.w700)));
      } else {
        spans.add(TextSpan(
            text: token.substring(1, token.length - 1),
            style: GoogleFonts.jetBrainsMono(
                fontSize: (base.fontSize ?? 15) - 1.5,
                color: const Color(0xFF7EE0C3),
                backgroundColor: RuiColors.surfaceHigh)));
      }
      cursor = m.end;
    }
    if (cursor < line.length) {
      spans.add(TextSpan(text: line.substring(cursor), style: base));
    }
    return spans;
  }

  @override
  Widget build(BuildContext context) {
    final base = GoogleFonts.inter(
        fontSize: 15, height: 1.5, color: RuiColors.text);
    final widgets = <Widget>[];
    final parts = text.split('```');

    for (var i = 0; i < parts.length; i++) {
      final part = parts[i];
      if (i.isOdd) {
        var code = part;
        final firstNewline = code.indexOf('\n');
        if (firstNewline != -1 && firstNewline < 20) {
          code = code.substring(firstNewline + 1);
        }
        widgets.add(Container(
          width: double.infinity,
          margin: const EdgeInsets.symmetric(vertical: 6),
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(
              color: const Color(0xFF0B0B0C),
              borderRadius: BorderRadius.circular(10),
              border: Border.all(color: const Color(0xFF2A2A2D))),
          child: SingleChildScrollView(
            scrollDirection: Axis.horizontal,
            child: SelectableText(code.trimRight(),
                style: GoogleFonts.jetBrainsMono(
                    fontSize: 12.5, color: const Color(0xFFD4D4D8))),
          ),
        ));
        continue;
      }
      for (final line in part.split('\n')) {
        if (line.trim().isEmpty) {
          widgets.add(const SizedBox(height: 6));
        } else if (line.startsWith('### ') ||
            line.startsWith('## ') ||
            line.startsWith('# ')) {
          final content = line.replaceFirst(RegExp(r'^#+\s'), '');
          widgets.add(Padding(
            padding: const EdgeInsets.only(top: 6, bottom: 2),
            child: SelectableText.rich(TextSpan(
                children: _spans(content,
                    base.copyWith(fontSize: 17, fontWeight: FontWeight.w700)))),
          ));
        } else if (RegExp(r'^\s*[-*]\s+').hasMatch(line)) {
          final content = line.replaceFirst(RegExp(r'^\s*[-*]\s+'), '');
          widgets.add(Padding(
            padding: const EdgeInsets.only(left: 6, top: 1, bottom: 1),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('\u2022  ', style: base),
                Expanded(
                    child: SelectableText.rich(
                        TextSpan(children: _spans(content, base)))),
              ],
            ),
          ));
        } else {
          widgets.add(
              SelectableText.rich(TextSpan(children: _spans(line, base))));
        }
      }
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      mainAxisSize: MainAxisSize.min,
      children: widgets,
    );
  }
}
