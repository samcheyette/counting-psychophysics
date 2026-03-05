#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <queue>
#include <unordered_map>
#include <type_traits>
#include <vector>

#if __cplusplus >= 202002L
namespace std {
template <class>
struct result_of;

template <class F, class... Args>
struct result_of<F(Args...)> : invoke_result<F, Args...> {};
} // namespace std
#endif

#include "Grammar.h"
#include "Singleton.h"
#include "DeterministicLOTHypothesis.h"
#include "TopN.h"
#include "MCMCChain.h"
#include "ParallelTempering.h"
#include "FullLZEnumeration.h"
#include "BasicEnumeration.h"
#include "PartialLZEnumeration.h"
#include "Fleet.h"

struct BinaryImage {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> pixels;

    BinaryImage() = default;

    BinaryImage(int w, int h)
        : width(w), height(h), pixels(static_cast<size_t>(w) * static_cast<size_t>(h), 0) {}

    bool operator==(const BinaryImage& other) const {
        return width == other.width && height == other.height && pixels == other.pixels;
    }

    bool in_bounds(int x, int y) const {
        return x >= 0 && y >= 0 && x < width && y < height;
    }

    size_t index(int x, int y) const {
        return static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
    }

    void set(int x, int y, uint8_t value = 1) {
        if (in_bounds(x, y)) {
            pixels[index(x, y)] = value;
        }
    }

    uint8_t get(int x, int y) const {
        return in_bounds(x, y) ? pixels[index(x, y)] : 0;
    }

    size_t mismatch_count(const BinaryImage& other) const {
        if (width != other.width || height != other.height) {
            throw std::runtime_error("Image dimensions must match for comparison");
        }
        size_t mismatches = 0;
        for (size_t i = 0; i < pixels.size(); ++i) {
            if (pixels[i] != other.pixels[i]) {
                ++mismatches;
            }
        }
        return mismatches;
    }

    void write_pgm(const std::string& path) const {
        std::ofstream out(path);
        if (!out) {
            throw std::runtime_error("Could not open output path: " + path);
        }
        out << "P2\n" << width << " " << height << "\n255\n";
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                out << (get(x, y) ? 0 : 255) << (x + 1 == width ? '\n' : ' ');
            }
        }
    }
};

std::ostream& operator<<(std::ostream& os, const BinaryImage& img) {
    os << "BinaryImage(" << img.width << "x" << img.height << ")";
    return os;
}

enum class JumpLocation : uint8_t {
    Center,
    TopLeft,
    Top,
    TopRight,
    Left,
    Right,
    BottomLeft,
    Bottom,
    BottomRight
};

struct TurtleInstruction {
    enum class Op : uint8_t { MoveLeft, MoveRight, MoveUp, MoveDown, PenUp, PenDown, Jump, JumpXY };
    Op op = Op::MoveRight;
    JumpLocation jump_location = JumpLocation::Center;
    int jump_x_idx = 0;
    int jump_y_idx = 0;

    bool operator==(const TurtleInstruction& other) const {
        return op == other.op &&
               jump_location == other.jump_location &&
               jump_x_idx == other.jump_x_idx &&
               jump_y_idx == other.jump_y_idx;
    }
};

struct TurtleProgram {
    std::vector<TurtleInstruction> ops;

    bool operator==(const TurtleProgram& other) const {
        return ops == other.ops;
    }
};

std::ostream& operator<<(std::ostream& os, const TurtleProgram& p) {
    os << "TurtleProgram(len=" << p.ops.size() << ")";
    return os;
}

int canvas_width = 200;
int canvas_height = 200;
double forward_step = 6.0;
double noise_epsilon = 0.05;
double dt_weight = 12.0;
double f1_weight = 220.0;
double max_temperature = 12.0;
std::string noise_model = "bernoulli+dt";
std::string mode = "infer";
std::string target_image_path = "";
size_t prior_tests = 10;
unsigned long prior_test_steps = 3000;
unsigned long enum_steps = 5000;
std::string enum_method = "full";
std::string program_format = "nested";
std::string export_prefix = "";
std::string trace_path = "";
unsigned long trace_every = 1;
double start_x = -1.0;
double start_y = -1.0;
int start_pen_down = 1;
int jump_grid = 16;
size_t show_top = 20;

static TurtleProgram one_op(TurtleInstruction::Op op) {
    TurtleProgram p;
    p.ops.push_back(TurtleInstruction{op, JumpLocation::Center});
    return p;
}

static TurtleProgram jump_to(JumpLocation loc) {
    TurtleProgram p;
    p.ops.push_back(TurtleInstruction{TurtleInstruction::Op::Jump, loc});
    return p;
}

static TurtleProgram jump_to_xy_idx(int x_idx, int y_idx) {
    TurtleProgram p;
    TurtleInstruction ins;
    ins.op = TurtleInstruction::Op::JumpXY;
    ins.jump_x_idx = x_idx;
    ins.jump_y_idx = y_idx;
    p.ops.push_back(ins);
    return p;
}

static TurtleProgram concat_prog(TurtleProgram a, const TurtleProgram& b) {
    a.ops.insert(a.ops.end(), b.ops.begin(), b.ops.end());
    return a;
}

static TurtleProgram repeat_prog(const TurtleProgram& p, int n) {
    TurtleProgram out;
    out.ops.reserve(p.ops.size() * static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        out.ops.insert(out.ops.end(), p.ops.begin(), p.ops.end());
    }
    return out;
}

static bool instruction_same_action(const TurtleInstruction& a, const TurtleInstruction& b) {
    return a.op == b.op &&
           a.jump_location == b.jump_location &&
           a.jump_x_idx == b.jump_x_idx &&
           a.jump_y_idx == b.jump_y_idx;
}

static std::vector<TurtleInstruction> normalize_instructions(const std::vector<TurtleInstruction>& in) {
    std::vector<TurtleInstruction> out;
    out.reserve(in.size());

    for (const auto& ins : in) {
        if (!out.empty()) {
            const auto& prev = out.back();

            if ((ins.op == TurtleInstruction::Op::PenUp && prev.op == TurtleInstruction::Op::PenUp) ||
                (ins.op == TurtleInstruction::Op::PenDown && prev.op == TurtleInstruction::Op::PenDown)) {
                continue;
            }

            if (ins.op == TurtleInstruction::Op::Jump && prev.op == TurtleInstruction::Op::Jump &&
                ins.jump_location == prev.jump_location) {
                continue;
            }
            if (ins.op == TurtleInstruction::Op::JumpXY && prev.op == TurtleInstruction::Op::JumpXY &&
                ins.jump_x_idx == prev.jump_x_idx && ins.jump_y_idx == prev.jump_y_idx) {
                continue;
            }
        }
        out.push_back(ins);
    }

    if (out.empty()) {
        out.push_back(TurtleInstruction{TurtleInstruction::Op::PenDown, JumpLocation::Center});
    }
    return out;
}

static std::string instruction_atom_expr(const TurtleInstruction& ins) {
    auto idx_tok = [](int v) {
        const int x = std::clamp(v, 0, 15);
        if (x < 10) {
            return str(x);
        }
        return std::string(1, static_cast<char>('a' + (x - 10)));
    };

    switch (ins.op) {
    case TurtleInstruction::Op::MoveLeft: return "left";
    case TurtleInstruction::Op::MoveRight: return "right";
    case TurtleInstruction::Op::MoveUp: return "up";
    case TurtleInstruction::Op::MoveDown: return "down";
    case TurtleInstruction::Op::PenUp: return "penup";
    case TurtleInstruction::Op::PenDown: return "pendown";
    case TurtleInstruction::Op::Jump:
        switch (ins.jump_location) {
        case JumpLocation::Center: return "jump(center)";
        case JumpLocation::TopLeft: return "jump(top_left)";
        case JumpLocation::Top: return "jump(top)";
        case JumpLocation::TopRight: return "jump(top_right)";
        case JumpLocation::Left: return "jump(left_edge)";
        case JumpLocation::Right: return "jump(right_edge)";
        case JumpLocation::BottomLeft: return "jump(bottom_left)";
        case JumpLocation::Bottom: return "jump(bottom)";
        case JumpLocation::BottomRight: return "jump(bottom_right)";
        }
        return "jump(center)";
    case TurtleInstruction::Op::JumpXY:
        return "jumpxy(" + idx_tok(ins.jump_x_idx) + "," + idx_tok(ins.jump_y_idx) + ")";
    }
    return "pendown";
}

static std::string program_expr_from_instructions(const std::vector<TurtleInstruction>& raw_ins) {
    auto ins = normalize_instructions(raw_ins);
    if (ins.empty()) {
        return "pendown";
    }

    std::vector<std::string> terms;
    terms.reserve(ins.size());

    for (size_t i = 0; i < ins.size();) {
        size_t j = i + 1;
        while (j < ins.size() && instruction_same_action(ins[i], ins[j])) {
            ++j;
        }
        size_t run = j - i;
        while (run > 0) {
            const size_t chunk = std::min<size_t>(100, run);
            const std::string atom = instruction_atom_expr(ins[i]);
            if (chunk == 1) {
                terms.push_back(atom);
            } else {
                terms.push_back("repeat" + str(chunk) + "(" + atom + ")");
            }
            run -= chunk;
        }
        i = j;
    }

    if (terms.size() == 1) {
        return terms[0];
    }

    std::string expr = terms[0];
    for (size_t i = 1; i < terms.size(); ++i) {
        expr = "seq(" + expr + "," + terms[i] + ")";
    }
    return expr;
}

static void draw_line(BinaryImage& img, double x0, double y0, double x1, double y1) {
    int x = static_cast<int>(std::lround(x0));
    int y = static_cast<int>(std::lround(y0));
    int x_end = static_cast<int>(std::lround(x1));
    int y_end = static_cast<int>(std::lround(y1));

    int dx = std::abs(x_end - x);
    int sx = x < x_end ? 1 : -1;
    int dy = -std::abs(y_end - y);
    int sy = y < y_end ? 1 : -1;
    int err = dx + dy;

    while (true) {
        img.set(x, y, 1);
        if (x == x_end && y == y_end) {
            break;
        }
        int e2 = 2 * err;
        if (e2 >= dy) {
            err += dy;
            x += sx;
        }
        if (e2 <= dx) {
            err += dx;
            y += sy;
        }
    }
}

static std::pair<double, double> resolve_jump_location(JumpLocation loc) {
    const double left = 0.0;
    const double right = static_cast<double>(canvas_width - 1);
    const double top = 0.0;
    const double bottom = static_cast<double>(canvas_height - 1);
    const double cx = (canvas_width - 1) / 2.0;
    const double cy = (canvas_height - 1) / 2.0;

    switch (loc) {
    case JumpLocation::Center: return {cx, cy};
    case JumpLocation::TopLeft: return {left, top};
    case JumpLocation::Top: return {cx, top};
    case JumpLocation::TopRight: return {right, top};
    case JumpLocation::Left: return {left, cy};
    case JumpLocation::Right: return {right, cy};
    case JumpLocation::BottomLeft: return {left, bottom};
    case JumpLocation::Bottom: return {cx, bottom};
    case JumpLocation::BottomRight: return {right, bottom};
    }
    return {cx, cy};
}

static std::pair<double, double> resolve_jump_xy_idx(int x_idx, int y_idx) {
    const int grid = std::max(2, jump_grid);
    const int gx = ((x_idx % grid) + grid) % grid;
    const int gy = ((y_idx % grid) + grid) % grid;
    const double px = (grid == 1) ? 0.0 : (static_cast<double>(gx) / static_cast<double>(grid - 1)) * static_cast<double>(canvas_width - 1);
    const double py = (grid == 1) ? 0.0 : (static_cast<double>(gy) / static_cast<double>(grid - 1)) * static_cast<double>(canvas_height - 1);
    return {px, py};
}

static BinaryImage render_program(const TurtleProgram& program) {
    BinaryImage img(canvas_width, canvas_height);

    double x = (start_x >= 0.0) ? start_x : (canvas_width - 1) / 2.0;
    double y = (start_y >= 0.0) ? start_y : (canvas_height - 1) / 2.0;
    bool pen_down = (start_pen_down != 0);

    for (const auto& instruction : program.ops) {
        if (instruction.op == TurtleInstruction::Op::MoveLeft) {
            const double nx = x - forward_step;
            const double ny = y;
            if (pen_down) {
                draw_line(img, x, y, nx, ny);
            }
            x = nx;
            y = ny;
        } else if (instruction.op == TurtleInstruction::Op::MoveRight) {
            const double nx = x + forward_step;
            const double ny = y;
            if (pen_down) {
                draw_line(img, x, y, nx, ny);
            }
            x = nx;
            y = ny;
        } else if (instruction.op == TurtleInstruction::Op::MoveUp) {
            const double nx = x;
            const double ny = y - forward_step;
            if (pen_down) {
                draw_line(img, x, y, nx, ny);
            }
            x = nx;
            y = ny;
        } else if (instruction.op == TurtleInstruction::Op::MoveDown) {
            const double nx = x;
            const double ny = y + forward_step;
            if (pen_down) {
                draw_line(img, x, y, nx, ny);
            }
            x = nx;
            y = ny;
        } else if (instruction.op == TurtleInstruction::Op::PenUp) {
            pen_down = false;
        } else if (instruction.op == TurtleInstruction::Op::PenDown) {
            pen_down = true;
        } else if (instruction.op == TurtleInstruction::Op::Jump) {
            const auto [jx, jy] = resolve_jump_location(instruction.jump_location);
            x = jx;
            y = jy;
        } else if (instruction.op == TurtleInstruction::Op::JumpXY) {
            const auto [jx, jy] = resolve_jump_xy_idx(instruction.jump_x_idx, instruction.jump_y_idx);
            x = jx;
            y = jy;
        }
    }

    return img;
}

static std::string cleaned_binary_row(const std::string& line) {
    std::string row;
    row.reserve(line.size());
    for (char c : line) {
        if (c == '0' || c == '1') {
            row.push_back(c);
        }
    }
    return row;
}

static BinaryImage load_binary_image_text(const std::string& path, int width_hint, int height_hint) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Could not open image file: " + path);
    }

    std::vector<std::string> rows;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::string row = cleaned_binary_row(line);
        if (!row.empty()) {
            rows.push_back(std::move(row));
        }
    }

    if (rows.empty()) {
        throw std::runtime_error("No binary rows found in image file: " + path);
    }

    const int inferred_height = static_cast<int>(rows.size());
    const int inferred_width = static_cast<int>(rows.front().size());

    for (const auto& row : rows) {
        if (static_cast<int>(row.size()) != inferred_width) {
            throw std::runtime_error("Image file has ragged rows: " + path);
        }
    }

    const int width = width_hint > 0 ? width_hint : inferred_width;
    const int height = height_hint > 0 ? height_hint : inferred_height;

    if (width != inferred_width || height != inferred_height) {
        throw std::runtime_error(
            "Input image dimensions are " + str(inferred_width) + "x" + str(inferred_height) +
            " but canvas is " + str(width) + "x" + str(height) +
            ". Set --width/--height to match or provide a matching file.");
    }

    BinaryImage img(width, height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            img.set(x, y, rows[static_cast<size_t>(y)][static_cast<size_t>(x)] == '1' ? 1 : 0);
        }
    }
    return img;
}

static double image_log_likelihood(const BinaryImage& predicted, const BinaryImage& target) {
    auto bernoulli_ll = [&]() {
    const double eps = std::clamp(noise_epsilon, 1e-9, 1.0 - 1e-9);
    const size_t mismatches = predicted.mismatch_count(target);
    const size_t total = static_cast<size_t>(predicted.width) * static_cast<size_t>(predicted.height);
    const size_t matches = total - mismatches;
    return static_cast<double>(matches) * std::log1p(-eps) +
           static_cast<double>(mismatches) * std::log(eps);
    };

    auto distance_transform = [](const BinaryImage& img) {
        const int w = img.width;
        const int h = img.height;
        const int inf = w + h + 5;
        std::vector<int> dist(static_cast<size_t>(w) * static_cast<size_t>(h), inf);
        std::queue<std::pair<int, int>> q;

        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                if (img.get(x, y)) {
                    dist[static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x)] = 0;
                    q.push({x, y});
                }
            }
        }

        if (q.empty()) {
            return dist;
        }

        while (!q.empty()) {
            auto [x, y] = q.front();
            q.pop();
            const int base = dist[static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x)];
            const int nx[4] = {x - 1, x + 1, x, x};
            const int ny[4] = {y, y, y - 1, y + 1};
            for (int i = 0; i < 4; ++i) {
                if (nx[i] < 0 || ny[i] < 0 || nx[i] >= w || ny[i] >= h) {
                    continue;
                }
                size_t idx = static_cast<size_t>(ny[i]) * static_cast<size_t>(w) + static_cast<size_t>(nx[i]);
                if (dist[idx] > base + 1) {
                    dist[idx] = base + 1;
                    q.push({nx[i], ny[i]});
                }
            }
        }
        return dist;
    };

    auto dt_ll = [&]() {
        const auto dist_to_target = distance_transform(target);
        const auto dist_to_pred = distance_transform(predicted);
        const int w = predicted.width;
        const int h = predicted.height;

        double sum_pred_to_target = 0.0;
        double sum_target_to_pred = 0.0;
        size_t pred_on = 0;
        size_t target_on = 0;

        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x);
                if (predicted.get(x, y)) {
                    sum_pred_to_target += static_cast<double>(dist_to_target[idx]);
                    ++pred_on;
                }
                if (target.get(x, y)) {
                    sum_target_to_pred += static_cast<double>(dist_to_pred[idx]);
                    ++target_on;
                }
            }
        }

        const double avg_pred = sum_pred_to_target / static_cast<double>(std::max<size_t>(1, pred_on));
        const double avg_target = sum_target_to_pred / static_cast<double>(std::max<size_t>(1, target_on));
        const double symmetric_avg = 0.5 * (avg_pred + avg_target);
        return -dt_weight * symmetric_avg;
    };

    auto f1_ll = [&]() {
        size_t tp = 0;
        size_t fp = 0;
        size_t fn = 0;

        const int w = predicted.width;
        const int h = predicted.height;
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const bool p = predicted.get(x, y) != 0;
                const bool t = target.get(x, y) != 0;
                if (p && t) {
                    ++tp;
                } else if (p && !t) {
                    ++fp;
                } else if (!p && t) {
                    ++fn;
                }
            }
        }

        const double denom = (2.0 * static_cast<double>(tp)) +
                             static_cast<double>(fp) +
                             static_cast<double>(fn);
        const double f1 = (denom > 0.0) ? ((2.0 * static_cast<double>(tp)) / denom) : 0.0;
        const double clamped_f1 = std::clamp(f1, 1e-12, 1.0);
        return f1_weight * std::log(clamped_f1);
    };

    if (noise_model == "bernoulli") {
        return bernoulli_ll();
    }
    if (noise_model == "dt") {
        return dt_ll();
    }
    if (noise_model == "f1") {
        return f1_ll();
    }
    if (noise_model == "bernoulli+dt") {
        return bernoulli_ll() + dt_ll();
    }
    if (noise_model == "f1+dt") {
        return f1_ll() + dt_ll();
    }

    throw std::runtime_error("Unsupported noise model: " + noise_model +
                             ". Supported: bernoulli, dt, f1, bernoulli+dt, f1+dt");
}

class TurtleGrammar : public Grammar<int, TurtleProgram, TurtleProgram, JumpLocation, int>,
                     public Singleton<TurtleGrammar> {
public:
    TurtleGrammar() {
        add("left", +[]() -> TurtleProgram { return one_op(TurtleInstruction::Op::MoveLeft); }, 3.0);
        add("right", +[]() -> TurtleProgram { return one_op(TurtleInstruction::Op::MoveRight); }, 3.0);
        add("up", +[]() -> TurtleProgram { return one_op(TurtleInstruction::Op::MoveUp); }, 3.0);
        add("down", +[]() -> TurtleProgram { return one_op(TurtleInstruction::Op::MoveDown); }, 3.0);
        add("penup", +[]() -> TurtleProgram { return one_op(TurtleInstruction::Op::PenUp); }, 1.0);
        add("pendown", +[]() -> TurtleProgram { return one_op(TurtleInstruction::Op::PenDown); }, 1.0);

        add("center", +[]() -> JumpLocation { return JumpLocation::Center; }, 1.0);
        add("top_left", +[]() -> JumpLocation { return JumpLocation::TopLeft; }, 1.0);
        add("top", +[]() -> JumpLocation { return JumpLocation::Top; }, 1.0);
        add("top_right", +[]() -> JumpLocation { return JumpLocation::TopRight; }, 1.0);
        add("left_edge", +[]() -> JumpLocation { return JumpLocation::Left; }, 1.0);
        add("right_edge", +[]() -> JumpLocation { return JumpLocation::Right; }, 1.0);
        add("bottom_left", +[]() -> JumpLocation { return JumpLocation::BottomLeft; }, 1.0);
        add("bottom", +[]() -> JumpLocation { return JumpLocation::Bottom; }, 1.0);
        add("bottom_right", +[]() -> JumpLocation { return JumpLocation::BottomRight; }, 1.0);

        add("jump(%s)", +[](JumpLocation loc) -> TurtleProgram { return jump_to(loc); }, 2.0);
        add("jumpxy(%s,%s)", +[](int x_idx, int y_idx) -> TurtleProgram {
            return jump_to_xy_idx(x_idx, y_idx);
        }, 2.0);

        for (int i = 0; i <= 15; ++i) {
            const double w = (i < 4) ? 4.0 : ((i < 8) ? 2.0 : 1.0);
            if (i < 10) {
                add_terminal(str(i), i, w);
            } else {
                add_terminal(std::string(1, static_cast<char>('a' + (i - 10))), i, w);
            }
        }

        add("seq(%s,%s)", +[](TurtleProgram a, TurtleProgram b) -> TurtleProgram {
            return concat_prog(std::move(a), b);
        }, 1.0);
        add("concat(%s,%s)", +[](TurtleProgram a, TurtleProgram b) -> TurtleProgram {
            return concat_prog(std::move(a), b);
        }, 1.0);

        for (int n = 2; n <= 100; ++n) {
            std::function<TurtleProgram(TurtleProgram)> f =
                [n](TurtleProgram p) -> TurtleProgram { return repeat_prog(p, n); };
            add("repeat" + str(n) + "(%s)",
                f,
                0.25);
        }
    }
} grammar;

using ImageDatum = defaultdatum_t<int, BinaryImage>;
using ImageData = std::span<ImageDatum>;

class TurtleHypothesis final
    : public DeterministicLOTHypothesis<
          TurtleHypothesis,
          int,
          TurtleProgram,
          TurtleGrammar,
          &grammar,
          ImageDatum,
          ImageData> {
public:
    using Super = DeterministicLOTHypothesis<
        TurtleHypothesis,
        int,
        TurtleProgram,
        TurtleGrammar,
        &grammar,
        ImageDatum,
        ImageData>;
    using Super::Super;

    std::string canonical_signature() const {
        auto* self = const_cast<TurtleHypothesis*>(this);
        auto prog = self->call(0, TurtleProgram{});
        return program_expr_from_instructions(prog.ops);
    }

    size_t hash() const override {
        return std::hash<std::string>{}(canonical_signature());
    }

    bool operator==(const TurtleHypothesis& other) const override {
        return canonical_signature() == other.canonical_signature();
    }

    [[nodiscard]] typename Super::ProposalType propose() const override {
        auto* self = const_cast<TurtleHypothesis*>(this);
        auto current_prog = normalize_instructions(self->call(0, TurtleProgram{}).ops);

        auto score_candidate = [&](const TurtleHypothesis& h) {
            auto cand_prog = normalize_instructions(const_cast<TurtleHypothesis&>(h).call(0, TurtleProgram{}).ops);
            const int len_delta = static_cast<int>(cand_prog.size()) - static_cast<int>(current_prog.size());

            int jumpxy_count = 0;
            int repeat_like = 0;
            for (const auto& ins : cand_prog) {
                if (ins.op == TurtleInstruction::Op::JumpXY) {
                    ++jumpxy_count;
                }
            }
            for (size_t i = 1; i < cand_prog.size(); ++i) {
                if (instruction_same_action(cand_prog[i - 1], cand_prog[i])) {
                    ++repeat_like;
                }
            }

            double score = 0.0;
            score += 0.6 * static_cast<double>(jumpxy_count);
            score += 0.25 * static_cast<double>(repeat_like);
            score += 0.2 * static_cast<double>(std::abs(len_delta) <= 2 ? 1 : 0);
            score += 0.05 * static_cast<double>(uniform());
            return score;
        };

        bool found = false;
        typename Super::ProposalType best{};
        double best_score = -infinity;

        for (int t = 0; t < 6; ++t) {
            auto p = Super::propose();
            if (!p) {
                continue;
            }
            auto [cand, fb] = p.value();
            const double s = score_candidate(cand);
            if (!found || s > best_score) {
                found = true;
                best_score = s;
                best = std::make_pair(cand, fb);
            }
        }

        if (found) {
            return best;
        }
        return Super::propose();
    }

    double compute_prior() override {
        double p = Super::compute_prior();
        if (p == -infinity) {
            return p;
        }

        auto* self = const_cast<TurtleHypothesis*>(this);
        auto prog = self->call(0, TurtleProgram{});
        auto ins = normalize_instructions(prog.ops);

        size_t penalty = 0;
        for (size_t i = 1; i < ins.size(); ++i) {
            if ((ins[i - 1].op == TurtleInstruction::Op::PenUp && ins[i].op == TurtleInstruction::Op::PenUp) ||
                (ins[i - 1].op == TurtleInstruction::Op::PenDown && ins[i].op == TurtleInstruction::Op::PenDown)) {
                ++penalty;
            }
            if ((ins[i - 1].op == TurtleInstruction::Op::Jump && ins[i].op == TurtleInstruction::Op::Jump &&
                 ins[i - 1].jump_location == ins[i].jump_location) ||
                (ins[i - 1].op == TurtleInstruction::Op::JumpXY && ins[i].op == TurtleInstruction::Op::JumpXY &&
                 ins[i - 1].jump_x_idx == ins[i].jump_x_idx && ins[i - 1].jump_y_idx == ins[i].jump_y_idx)) {
                ++penalty;
            }
        }

        if (penalty > 0) {
            p -= static_cast<double>(penalty) * 2.5;
        }
        return this->prior = p;
    }

    double compute_single_likelihood(const datum_t& datum) override {
        const auto program = call(datum.input, TurtleProgram{});
        const auto predicted = render_program(program);
        return datum.count * image_log_likelihood(predicted, datum.output);
    }
};

struct InferenceResult {
    TopN<TurtleHypothesis> top;
    std::vector<ImageDatum> data_storage;
};

static std::string imperative_program_string(const std::string& fleet_program);

static void print_progress_line(
    const std::string& label,
    unsigned long current,
    unsigned long total,
    const std::chrono::steady_clock::time_point& t0) {
    const auto now = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(now - t0).count();
    const double frac = (total > 0) ? (static_cast<double>(current) / static_cast<double>(total)) : 0.0;
    const int pct = static_cast<int>(std::clamp(frac, 0.0, 1.0) * 100.0);
    const double eta = (current > 0 && total > current)
        ? elapsed * (static_cast<double>(total - current) / static_cast<double>(current))
        : 0.0;
    print("# progress", label, str(current) + "/" + str(total), str(pct) + "%", "elapsed_s", elapsed, "eta_s", eta);
}

static InferenceResult run_mcmc(const BinaryImage& target, unsigned long steps_override = 0, unsigned long runtime_override = 0) {
    InferenceResult out;
    out.top = TopN<TurtleHypothesis>(FleetArgs::ntop);
    out.data_storage.emplace_back(0, target, NaN, 1.0);

    ImageData data_view(out.data_storage.data(), out.data_storage.size());

    auto h0 = TurtleHypothesis::sample();
    ParallelTempering<TurtleHypothesis> sampler(h0, data_view, std::max<unsigned long>(1, FleetArgs::nchains), max_temperature);

    unsigned long ctl_steps = steps_override > 0 ? steps_override : FleetArgs::steps;
    unsigned long ctl_runtime = runtime_override > 0 ? runtime_override : FleetArgs::runtime;
    if (ctl_steps > 0) {
        ctl_runtime = 0;
    }

    Control ctl(ctl_steps, ctl_runtime, 1, FleetArgs::restart);

    const auto t0 = std::chrono::steady_clock::now();
    unsigned long iter = 0;
    unsigned long next_pct_mark = 10;
    std::ofstream trace_out;
    if (!trace_path.empty()) {
        trace_out.open(trace_path);
        if (!trace_out) {
            throw std::runtime_error("Could not open trace file: " + trace_path);
        }
        trace_out << "sample\tposterior\tprior\tlikelihood\tprogram\n";
    }

    for (auto& h : sampler.run(ctl) | thin(FleetArgs::thin) | printer(FleetArgs::print)) {
        ++iter;
        out.top << h;
        if (trace_out && (iter % std::max<unsigned long>(1, trace_every) == 0)) {
            trace_out << iter << '\t' << h.posterior << '\t' << h.prior << '\t' << h.likelihood
                      << '\t' << imperative_program_string(h.string()) << '\n';
        }
        if (ctl_steps > 0) {
            const unsigned long pct_now = (iter * 100UL) / std::max<unsigned long>(1, ctl_steps);
            if (pct_now >= next_pct_mark || iter == ctl_steps) {
                print_progress_line("infer", std::min(iter, ctl_steps), ctl_steps, t0);
                next_pct_mark += 10;
            }
        }
    }

    return out;
}

template <typename EnumeratorT>
static InferenceResult run_enumeration_with(const BinaryImage& target, unsigned long enum_steps_override = 0) {
    InferenceResult out;
    out.top = TopN<TurtleHypothesis>(FleetArgs::ntop);
    out.data_storage.emplace_back(0, target, NaN, 1.0);
    ImageData data_view(out.data_storage.data(), out.data_storage.size());

    const unsigned long nsteps = enum_steps_override > 0 ? enum_steps_override : enum_steps;
    EnumeratorT enumerator(&grammar);
    const auto t0 = std::chrono::steady_clock::now();
    unsigned long next_pct_mark = 10;

    for (enumerationidx_t z = 0; z < nsteps && !CTRL_C; ++z) {
        try {
            auto n = enumerator.toNode(z, grammar.start());
            TurtleHypothesis h(n);
            h.compute_posterior(data_view);
            out.top << h;
        } catch (const std::exception&) {
        } catch (...) {
        }
        const unsigned long done = static_cast<unsigned long>(z + 1);
        const unsigned long pct_now = (done * 100UL) / std::max<unsigned long>(1, nsteps);
        if (pct_now >= next_pct_mark || done == nsteps) {
            print_progress_line("enumerate", done, nsteps, t0);
            next_pct_mark += 10;
        }
    }

    return out;
}

static InferenceResult run_enumeration(const BinaryImage& target, unsigned long enum_steps_override = 0) {
    if (enum_method == "full") {
        return run_enumeration_with<FullLZEnumeration<TurtleGrammar>>(target, enum_steps_override);
    }
    if (enum_method == "partial") {
        return run_enumeration_with<PartialLZEnumeration<TurtleGrammar>>(target, enum_steps_override);
    }
    if (enum_method == "basic") {
        return run_enumeration_with<BasicEnumeration<TurtleGrammar>>(target, enum_steps_override);
    }
    throw std::runtime_error("Unsupported --enum-method: " + enum_method + " (use basic, partial, or full)");
}

static std::string trim_copy(const std::string& s) {
    size_t b = 0;
    while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b]))) {
        ++b;
    }
    size_t e = s.size();
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) {
        --e;
    }
    return s.substr(b, e - b);
}

static std::string strip_lambda_prefix(const std::string& s) {
    const auto dot = s.find('.');
    if (dot == std::string::npos) {
        return trim_copy(s);
    }
    return trim_copy(s.substr(dot + 1));
}

static bool starts_with(const std::string& s, const std::string& prefix) {
    return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

static std::string indent_str(int n) {
    return std::string(static_cast<size_t>(std::max(0, n)), ' ');
}

struct ParsedExpr {
    bool is_call = false;
    std::string head;
    std::vector<ParsedExpr> args;
};

class ExprParser {
public:
    explicit ExprParser(const std::string& s_in) : s(trim_copy(s_in)) {}

    ParsedExpr parse_all() {
        ParsedExpr out = parse_expr();
        skip_ws();
        if (pos != s.size()) {
            throw std::runtime_error("Unexpected trailing text while parsing expression");
        }
        return out;
    }

private:
    const std::string s;
    size_t pos = 0;

    void skip_ws() {
        while (pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos]))) {
            ++pos;
        }
    }

    bool consume(char c) {
        skip_ws();
        if (pos < s.size() && s[pos] == c) {
            ++pos;
            return true;
        }
        return false;
    }

    std::string parse_token() {
        skip_ws();
        const size_t start = pos;
        while (pos < s.size()) {
            const char c = s[pos];
            if (std::isalnum(static_cast<unsigned char>(c)) || c == '_') {
                ++pos;
            } else {
                break;
            }
        }
        if (pos == start) {
            throw std::runtime_error("Expected token while parsing expression");
        }
        return s.substr(start, pos - start);
    }

    ParsedExpr parse_expr() {
        ParsedExpr node;
        node.head = parse_token();
        skip_ws();

        if (!consume('(')) {
            node.is_call = false;
            return node;
        }

        node.is_call = true;
        skip_ws();
        if (consume(')')) {
            return node;
        }

        while (true) {
            node.args.push_back(parse_expr());
            skip_ws();
            if (consume(',')) {
                continue;
            }
            if (consume(')')) {
                break;
            }
            throw std::runtime_error("Expected ',' or ')' while parsing expression");
        }
        return node;
    }
};

static std::string expr_to_source(const ParsedExpr& e) {
    if (!e.is_call) {
        return e.head;
    }

    std::string out = e.head + "(";
    for (size_t i = 0; i < e.args.size(); ++i) {
        if (i > 0) {
            out += ",";
        }
        out += expr_to_source(e.args[i]);
    }
    out += ")";
    return out;
}

static bool is_repeat_call(const ParsedExpr& e, long long& n_out) {
    if (!e.is_call || !starts_with(e.head, "repeat")) {
        return false;
    }
    const std::string nstr = e.head.substr(6);
    if (nstr.empty()) {
        return false;
    }
    if (!std::all_of(nstr.begin(), nstr.end(), [](char c) {
            return std::isdigit(static_cast<unsigned char>(c));
        })) {
        return false;
    }
    n_out = std::stoll(nstr);
    return true;
}

static int parse_jump_idx_atom(const ParsedExpr& e) {
    if (e.is_call) {
        throw std::runtime_error("jumpxy index must be an atom");
    }
    const std::string tok = trim_copy(e.head);
    if (tok.size() == 1) {
        const char c = tok[0];
        if (c >= '0' && c <= '9') {
            return static_cast<int>(c - '0');
        }
        if (c >= 'a' && c <= 'f') {
            return 10 + static_cast<int>(c - 'a');
        }
    }
    return atoi(tok.c_str());
}

static void imperative_lines_from_ast(const ParsedExpr& e, int indent, std::vector<std::string>& out, bool expand_repeats) {
    auto emit = [&](const std::string& line) {
        out.push_back(indent_str(indent) + line);
    };

    if (!e.is_call) {
        if (e.head == "left" || e.head == "right" || e.head == "up" || e.head == "down" ||
            e.head == "penup" || e.head == "pendown") {
            emit(e.head + "()");
        } else {
            emit(e.head);
        }
        return;
    }

    if ((e.head == "seq" || e.head == "concat") && e.args.size() == 2) {
        imperative_lines_from_ast(e.args[0], indent, out, expand_repeats);
        imperative_lines_from_ast(e.args[1], indent, out, expand_repeats);
        return;
    }

    long long nrep = 0;
    if (is_repeat_call(e, nrep) && e.args.size() == 1) {
        if (expand_repeats) {
            const long long k = std::max<long long>(0, nrep);
            for (long long i = 0; i < k; ++i) {
                imperative_lines_from_ast(e.args[0], indent, out, expand_repeats);
            }
            return;
        }
        emit("repeat " + str(nrep) + ":");
        imperative_lines_from_ast(e.args[0], indent + 4, out, expand_repeats);
        return;
    }

    if (e.head == "jump" && e.args.size() == 1 && !e.args[0].is_call) {
        emit("jump(" + e.args[0].head + ")");
        return;
    }

    if (e.head == "jumpxy" && e.args.size() == 2) {
        const int xi = parse_jump_idx_atom(e.args[0]);
        const int yi = parse_jump_idx_atom(e.args[1]);
        emit("jumpxy(" + str(xi) + "," + str(yi) + ")");
        return;
    }

    emit(expr_to_source(e));
}

static std::string imperative_program_string(const std::string& fleet_program) {
    const std::string expr = strip_lambda_prefix(fleet_program);
    std::string pretty;
    try {
        if (program_format != "nested" && program_format != "expanded") {
            throw std::runtime_error("Unsupported --program-format: " + program_format +
                                     " (use nested or expanded)");
        }
        const bool expand_repeats = (program_format == "expanded");
        ExprParser parser(expr);
        const ParsedExpr ast = parser.parse_all();
        std::vector<std::string> lines;
        imperative_lines_from_ast(ast, 0, lines, expand_repeats);
        for (size_t i = 0; i < lines.size(); ++i) {
            if (i > 0) {
                pretty.push_back('\n');
            }
            pretty += lines[i];
        }
    } catch (const std::exception&) {
        pretty = expr;
    }
    std::string escaped;
    escaped.reserve(pretty.size() + 16);
    for (char c : pretty) {
        if (c == '\n') {
            escaped += "\\n";
        } else {
            escaped.push_back(c);
        }
    }
    return escaped;
}

static void print_ranked_programs(const TopN<TurtleHypothesis>& top, const BinaryImage& target, size_t limit) {
    if (top.empty()) {
        print("# No hypotheses collected.");
        return;
    }

    const auto sorted = top.sorted(false);
    const size_t k = std::min(limit, sorted.size());
    const double logz = const_cast<TopN<TurtleHypothesis>&>(top).Z();

    print("# rank\tposterior\tprior\tlikelihood\tweight\tmismatches\tprogram");
    for (size_t i = 0; i < k; ++i) {
        auto h = sorted[i];
        const auto prog = h.call(0, TurtleProgram{});
        const auto pred = render_program(prog);
        const auto mismatches = pred.mismatch_count(target);
        const double weight = std::exp(h.posterior - logz);
        print(i + 1, h.posterior, h.prior, h.likelihood, weight, mismatches, imperative_program_string(h.string()));
    }
}

static void run_prior_predictive_tests() {
    print("# Running prior predictive tests", prior_tests);

    size_t recovered = 0;
    double mean_best_mismatch_rate = 0.0;

    for (size_t i = 0; i < prior_tests; ++i) {
        auto true_h = TurtleHypothesis::sample();
        const auto true_prog = true_h.call(0, TurtleProgram{});
        const auto target = render_program(true_prog);

        if (!export_prefix.empty()) {
            target.write_pgm(export_prefix + "_target_" + str(i) + ".pgm");
        }

        auto result = run_mcmc(target, prior_test_steps, 0);

        ImageData dv(result.data_storage.data(), result.data_storage.size());
        true_h.compute_posterior(dv);
        const bool found_true = result.top.contains(true_h);
        recovered += found_true ? 1 : 0;

        auto best = result.top.best();
        const auto best_prog = best.call(0, TurtleProgram{});
        const auto best_img = render_program(best_prog);
        const double mismatch_rate = static_cast<double>(best_img.mismatch_count(target)) /
                                     static_cast<double>(canvas_width * canvas_height);
        mean_best_mismatch_rate += mismatch_rate;

        print("# prior-test", i + 1, "recovered", found_true ? 1 : 0,
              "best_mismatch_rate", mismatch_rate,
              "true", true_h.string(),
              "best", best.string());
    }

    mean_best_mismatch_rate /= static_cast<double>(std::max<size_t>(1, prior_tests));

    print("# prior-test-summary",
          "recovered", recovered,
          "of", prior_tests,
          "recovery_rate", static_cast<double>(recovered) / static_cast<double>(std::max<size_t>(1, prior_tests)),
          "mean_best_mismatch_rate", mean_best_mismatch_rate);
}

int main(int argc, char** argv) {
    FleetArgs::steps = 20000;
    FleetArgs::nchains = 4;
    FleetArgs::ntop = 100;
    FleetArgs::timestring = "120s";

    Fleet fleet("Program induction for dot-pattern images (turtle primitives + MCMC)");

    fleet.add_option("--mode", mode, "infer or prior-test");
    fleet.add_option("--target-image", target_image_path, "Path to target image text file (0/1 raster)");
    fleet.add_option("--width", canvas_width, "Canvas width in pixels");
    fleet.add_option("--height", canvas_height, "Canvas height in pixels");
    fleet.add_option("--step-size", forward_step, "Turtle forward step size in pixels");
    fleet.add_option("--noise-model", noise_model, "Noise model: bernoulli, dt, f1, bernoulli+dt, or f1+dt");
    fleet.add_option("--epsilon", noise_epsilon, "Bernoulli pixel-flip noise probability");
    fleet.add_option("--dt-weight", dt_weight, "Weight for distance-transform shape term");
    fleet.add_option("--f1-weight", f1_weight, "Weight for foreground F1 log-likelihood term");
    fleet.add_option("--max-temp", max_temperature, "Maximum temperature for parallel tempering ladder");
    fleet.add_option("--prior-tests", prior_tests, "How many prior predictive tests to run");
    fleet.add_option("--prior-steps", prior_test_steps, "MCMC steps per prior-test inference run");
    fleet.add_option("--enum-steps", enum_steps, "How many hypotheses to enumerate in enumerate mode");
    fleet.add_option("--enum-method", enum_method, "Enumeration method: basic, partial, or full");
    fleet.add_option("--program-format", program_format, "Program print format: nested or expanded");
    fleet.add_option("--start-x", start_x, "Initial turtle x-coordinate (-1 means center)");
    fleet.add_option("--start-y", start_y, "Initial turtle y-coordinate (-1 means center)");
    fleet.add_option("--start-pen-down", start_pen_down, "1 starts with pen down, 0 starts pen up");
    fleet.add_option("--jump-grid", jump_grid, "Grid size used to map jumpxy(x,y) indices to pixels (4,8,16)");
    fleet.add_option("--show-top", show_top, "How many top programs to print");
    fleet.add_option("--export-prefix", export_prefix, "If set, write PGM outputs with this prefix");
    fleet.add_option("--trace-path", trace_path, "If set, write inference sample trace TSV to this path");
    fleet.add_option("--trace-every", trace_every, "Record every Nth sample in trace output");

    fleet.initialize(argc, argv);

    if (canvas_width <= 0 || canvas_height <= 0) {
        throw std::runtime_error("Canvas width and height must be > 0");
    }
    if (!(jump_grid == 4 || jump_grid == 8 || jump_grid == 16)) {
        throw std::runtime_error("--jump-grid must be one of: 4, 8, 16");
    }
    if (max_temperature < 1.0) {
        throw std::runtime_error("--max-temp must be >= 1");
    }

    if (mode == "prior-test") {
        run_prior_predictive_tests();
        return 0;
    }

    if (!(mode == "infer" || mode == "enumerate")) {
        throw std::runtime_error("Unsupported mode: " + mode + " (use infer, enumerate, or prior-test)");
    }

    if (target_image_path.empty()) {
        throw std::runtime_error("--target-image is required in infer mode");
    }

    const auto target = load_binary_image_text(target_image_path, canvas_width, canvas_height);
    auto result = (mode == "enumerate") ? run_enumeration(target) : run_mcmc(target);

    if (!export_prefix.empty() && !result.top.empty()) {
        auto best = result.top.best();
        const auto best_prog = best.call(0, TurtleProgram{});
        render_program(best_prog).write_pgm(export_prefix + "_best.pgm");
        target.write_pgm(export_prefix + "_target.pgm");
    }

    print_ranked_programs(result.top, target, show_top);

    return 0;
}
