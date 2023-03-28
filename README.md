# 令人敬畏的WebAssembly运行时 ![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)

本存储库包含执行WebAssembly（wasm）格式和/或将其编译为可执行机器代码的虚拟机和工具的列表：:octocat:

#### 图例
:rocket: - 活跃开发</br>
:sleeping: - 未维护（距上次提交已超过一年）</br>

## 目录

- [aWsm](#awsm) <sup><sup>:rocket:</sup></sup></br>
- [CMM of Wasm](#cmm) <sup><sup>:sleeping:</sup></sup></br>
- [EOSVM](#eosvm) <sup><sup>:rocket:</sup></sup></br>
- [FDVM](#fdvm) <sup><sup>:sleeping:</sup></sup></br>
- [Fizzy](#fizzy) <sup><sup>:rocket:</sup></sup></br>
- [GraalWASM](#graalwasm) <sup><sup>:rocket:</sup></sup></br>
- [Happy New Moon With Report](#happy-new-moon-with-report) <sup><sup>:rocket:</sup></sup></br>
- [inNative](#innative) <sup><sup>:rocket:</sup></sup></br>
- [生命](#生命) <sup><sup>:sleeping:</sup></sup></br>
- [Lucet](#lucet) <sup><sup>:rocket:</sup></sup></br>
- [wasm3](#wasm3) <sup><sup>:rocket:</sup></sup></br>
- [电机](#电机) <sup><sup>:sleeping:</sup></sup></br>
- [py-wasm](#py-wasm) <sup><sup>:sleeping:</sup></sup></br>
- [Serverless Wasm](#serverless-wasm) <sup><sup>:sleeping:</sup></sup></br>
- [Swam](#swam) <sup><sup>:rocket:</sup></sup></br>
- [VMIR](#vmir) <sup><sup>:sleeping:</sup></sup></br>
- [wac](#wac) <sup><sup>:sleeping:</sup></sup></br>
- [Wagon](#wagon) <sup><sup>:sleeping:</sup></sup></br>
- [WAKit](#wakit) <sup><sup>:rocket:</sup></sup></br>
- [Warpy](#warpy) <sup><sup>:sleeping:</sup></sup></br>
- [WasmEdge](#wasmedge) <sup><sup>:rocket:</sup></sup></br>
- [Wasmer](#wasmer) <sup><sup>:rocket:</sup></sup></br>
- [Wasmi](#wasmi) <sup><sup>:rocket:</sup></sup></br>
- [Wasmo](#wasmo) <sup><sup>:rocket:</sup></sup></br>
- [WasmRT](#wasmrt) <sup><sup>:sleeping:</sup></sup></br>
- [Wasmtime](#wasmtime) <sup><sup>:rocket:</sup></sup></br>
- [WasmVM](#wasmvm1) <sup><sup>:rocket:</sup></sup></br>- [WasmVM](#wasmvm2) <sup><sup>:sleeping:</sup></sup></br>
- [WAVM](#wavm) <sup><sup>:rocket:</sup></sup></br>
- [WebAssembly](#webassembly) <sup><sup>:sleeping:</sup></sup></br>
- [WebAssembly Micro Runtime](#wamr) <sup><sup>:rocket:</sup></sup></br>
- [TWVM](#twvm) <sup><sup>:rocket:</sup></sup></br>
- [wazero](#wazero) <sup><sup>:rocket:</sup></sup></br>

----------------


## <a name="awsm"></a>[aWsm](https://github.com/gwsystems/aWsm) <sup>[top⇈](#contents)</sup>

> aWsm 是一个编译器和运行时，可以将 WebAssembly（Wasm）代码编译成 llvm 字节码，然后编译成沙盒二进制文件在不同的平台上运行。

* **编写语言**

    <table>
    <tr>
        <td>Rust</td>
        <td>C</td>
    </tr>
    </table>

* **编译器框架**

    <table>
    <tr>
        <td>LLVM</td>
    </tr>
    </table>


* **编译/执行模式**

    <table>
    <tr>
        <td>AOT</td>
    </tr>
    </table>

* **与其他语言的互操作性**

    <table>
    <tr>
        <td>C</td>
    </tr>
    </table>

* **支持的 MVP 之外的特性**

    - `N/A`

* **支持的宿主 API**

    - `N/A`

* **非 Web 标准**

    - `N/A`

* **被以下项目使用**

    - `N/A`

* **支持的平台**

    <table>
    <tr>
        <td>Linux</td>
        <td>macOS</td>
    </tr>
    </table>


## <a name="cmm"></a>[Wasm 的 CMM](https://github.com/SimonJF/cmm_of_wasm) <sup>[top⇈](#contents)</sup>
> 一个从 WebAssembly 到本地代码的编译器，通过 OCaml 后端。

* **编写语言**

    <table>
    <tr>
        <td>OCaml</td>
    </tr>
    </table>

* **编译器框架**

    <table>
    <tr>
        <td>OCaml Compiler</td>
    </tr>
    </table>


* **编译/执行模式**

    <table>
    <tr>
        <td>AOT</td>
    </tr>
    </table>

* **与其他语言的互操作性**

    - `N/A`

* **支持的 MVP 之外的特性**

    - `N/A`

* **支持的宿主 API**

    - `N/A`

* **非 Web 标准**

    - `N/A`

* **被以下项目使用**

    - `N/A`

* **支持的平台**

    <table><tr>
        <td>Linux</td>
        <td>macOS</td>
    </tr>
    </table>

## <a name="eosvm"></a>[EOSVM](https://github.com/EOSIO/eos-vm) <sup>[返回顶部⇈](#contents)</sup>
> EOS VM是专门为区块链应用的高需求而设计，这些需求远远超出了为Web浏览器或标准开发而设计的WebAssembly引擎的要求。

* **使用的语言**

    <table>
    <tr>
        <td>C++</td>
    </tr>
    </table>

* **编译器框架**

    <table>
    <tr>
        <td>自定义</td>
    </tr>
    </table>


* **编译/执行模式**

    <table>
    <tr>
        <td>解释执行</td>
    </tr>
    </table>

* **与其他语言的互操作性**

    - `未适用`

* **支持的非MVP功能**

    - `未适用`

* **支持的主机API**

    - `未适用`

* **非Web标准**

    - `未适用`

* **使用情况**

    - `未适用`

* **支持的平台**

    <table>
    <tr>
        <td>Linux</td>
        <td>macOS</td>
        <td>Windows</td>
    </tr>
    </table>


## <a name="fdvm"></a>[FDVM](https://github.com/funcdef/fdvm) <sup>[返回顶部⇈](#contents)</sup>
> 用于开发服务器端WebAssembly应用的WASM运行时。

* **使用的语言**

    <table>
    <tr>
        <td>JavaScript</td>
    </tr>
    </table>

* **编译器框架**

    <table>
    <tr>
        <td>NodeJS(V8)</td>
    </tr>
    </table>


* **编译/执行模式**

    <table>
    <tr>
        <td>JIT</td>
    </tr>
    </table>

* **与其他语言的互操作性**

    - `未适用`

* **支持的非MVP功能**

    - `未适用`

* **支持的主机API**

    - `未适用`

* **非Web标准**

    - `未适用`

* **使用情况**

    - `未适用`

* **支持的平台**

    <table>
    <tr>
        <td>Linux</td>
        <td>macOS</td>
        <td>Windows</td>
    </tr>
    </table>


## <a name="fizzy"></a>[Fizzy](https://github.com/wasmx/fizzy) <sup>[返回顶部⇈](#contents)</sup>
> Fizzy旨在成为一个快速、确定性和严谨的C++ WebAssembly解释器。* **编写语言**

    <table>
    <tr>
        <td>C++</td>
    </tr>
    </table>

* **编译器框架**

    - `N/A`

* **编译 / 执行模式**

    <table>
    <tr>
        <td>解释器</td>
    </tr>
    </table>

* **与其他编程语言的互操作性**

    - `N/A`

* **支持的非MVP功能**

    -`N/A`

* **支持的主机API**

    <table>
    <tr>
        <td>WASI</td>
    </tr>
    </table>

* **非Web标准**

    <table>
    <tr>
        <td>WASI</td>
    </tr>
    </table>

* **使用者**

    - `N/A`

* **支持的平台**

    <table>
    <tr>
        <td>Linux</td>
        <td>macOS</td>
    </tr>
    </table>

## <a name="graalwasm"></a>[GraalWasm](https://github.com/oracle/graal/tree/master/wasm) <sup>[top⇈](#contents)</sup>

GraalWasm 是在 GraalVM中实现的 WebAssembly 引擎。它可以解释和编译二进制格式的WebAssembly程序，也可以嵌入到其他程序中。

* **编写语言**

    <table>
    <tr>
        <td>Java</td>
    </tr>
    </table>

* **编译器框架**

    <table>
    <tr>
        <td>GraalVM</td>
    </tr>
    </table>


* **编译 / 执行模式**

    <table>
    <tr>
        <td>JIT</td>
    </tr>
    </table>

* **与其他编程语言的互操作性**

    <table>
    <tr>
        <td>Java</td>
        <td>JVM</td>
        <td>Graal 支持的语言</td>
    </tr>
    </table>

* **支持的非MVP功能**

    <table>
    <tr>
    </tr>
    </table>

* **支持的主机API**

    <table>
    <tr>
        <td>WASI</td>
    </tr>
    </table>

* **非Web标准**

    <table>
    <tr>
        <td>WASI</td>
    </tr>
    </table>

* **使用者**

    - [GraalVM JavaScript](https://github.com/graalvm/graaljs) - JavaScript编程语言的高性能实现。

* **支持的平台**

    <table>
    <tr>
        <td>Linux</td>
        <td>macOS</td>
        <td>Windows</td>
    </tr>
    </table>## <a name="happy-new-moon-with-report"></a>[祝新月快乐与报告](https://github.com/fishjd/HappyNewMoonWithReport) <sup>[返回顶部⇈](#contents)</sup>

祝新月快乐与报告是一个完全使用Java编写的WebAssembly开源实现。它用于在Java中运行或测试Web Assembly模块（*.wasm）。

* **编写的语言**

    <table>
    <tr>
        <td>Java</td>
    </tr>
    </table>

* **编译器架构**

    - `N/A`

* **编译/执行模式**

    <table>
    <tr>
        <td>解释执行</td>
    </tr>
    </table>

* **与其他语言的互操作性**

    <table>
    <tr>
        <td>Java</td>
        <td>JVM语言</td>
    </tr>
    </table>

* **支持的非MVP功能**

    <table>
    <tr>
    	<td>符号扩展</td>
    </tr>
    </table>

* **支持的主机API**

    - `N/A`

* **被使用者**

    - `N/A`


* **支持的平台**

    <table>
    <tr>
        <td>JVM</td>
        <td>Linux</td>
        <td>macOS</td>
        <td>Windows</td>
    </tr>
    </table>

## <a name="innative"></a>[inNative](https://github.com/innative-sdk/innative) <sup>[返回顶部⇈](#contents)</sup>
> 一种用于WebAssembly的AOT(提前生成)编译器，它创建与C兼容的二进制文件，可以作为沙盒插件动态加载，也可以作为直接与操作系统接口的独立可执行文件。

* **编写的语言**

    <table>
    <tr>
        <td>C++</td>
    </tr>
    </table>

* **编译器架构**

    <table>
    <tr>
        <td>LLVM</td>
    </tr>
    </table>


* **编译/执行模式**

    <table>
    <tr>
        <td>AOT</td>
        <td>JIT</td>
    </tr>
    </table>

* **与其他语言的互操作性**

    - `N/A`

* **支持的非MVP功能**

    <table>
    <tr>
        <td>线程</td>
        <td>多个结果和块参数</td>
        <td>名称部分</td>
        <td>批量内存操作</td>
        <td>符号扩展指令</td><td>非陷阱转换指令</td>
    </tr>
    </table>

* **支持的宿主API**

    <table>
    <tr>
        <td>Custom</td>
    </tr>
    </table>

* **非Web标准**

    - `N/A`

* **使用者**

    - `N/A`

* **支持的平台**

    <table>
    <tr>
        <td>Linux</td>
        <td>macOS</td>
        <td>Windows</td>
    </tr>
    </table>


## <a name="life"></a>[Life](https://github.com/perlin-network/life) <sup>[top⇈](#contents)</sup>
> Life是一个为去中心化应用构建的安全快速WebAssembly虚拟机，由Perlin Network使用Go语言编写。

* **采用的语言**

    <table>
    <tr>
        <td>Go</td>
    </tr>
    </table>

* **编译器框架**

    <table>
    <tr>
        <td>Custom</td>
    </tr>
    </table>


* **编译/执行模式**

    <table>
    <tr>
        <td>解释性</td>
    </tr>
    </table>

* **与其他语言的互操作性**

    - `N/A`

* **支持的非MVP特性**

    - `N/A`

* **支持的宿主API**

    - `N/A`

* **非Web标准**

    - `N/A`

* **使用者**

    - `N/A`

* **支持的平台**

    <table>
    <tr>
        <td>Linux</td>
        <td>macOS</td>
        <td>Windows</td>
    </tr>
    </table>

## <a name="lucet"></a>[Lucet](https://github.com/fastly/lucet) <sup>[top⇈](#contents)</sup>
> Lucet是一种本地WebAssembly编译器和运行时。它被设计为在应用程序中安全地执行不受信任的WebAssembly程序。

> 注意：Lucet现在处于维护模式。工作已转移到[wasmtime](https://github.com/bytecodealliance/wasmtime/)。

* **采用的语言**

    <table>
    <tr>
        <td>Rust</td>
    </tr>
    </table>

* **编译器框架**

    <table>
    <tr>
        <td>Cranelift</td>
    </tr>
    </table>


* **编译/执行模式**

    <table>
    <tr>
        <td>AOT</td>
    </tr>
    </table>

* **与其他语言的互操作性**

    - `N/A`

* **支持的非MVP特性**

    - `N/A`

* **支持的宿主API**". 

    - `N/A`<table>
    <tr>
        <td>WASI</td>
    </tr>
</table>

* **非 Web 标准**

    <table>
    <tr>
        <td>WASI</td>
    </tr>
    </table>

* **使用者**

    - `N/A`

* **支持的平台**

    <table>
    <tr>
        <td>Linux</td>
    </tr>
    </table>


## <a name="wasm3"></a>[Wasm3](https://github.com/wasm3/wasm3) <sup>[返回顶部⇈](#contents)</sup>
> 🚀 一款高性能的 WebAssembly 解释器

* **使用的编程语言**

    <table>
    <tr>
        <td>C</td>
    </tr>
    </table>

* **编译框架**

    <table>
    <tr>
        <td>自定义</td>
    </tr>
    </table>


* **编译/执行模式**

    <table>
    <tr>
        <td>解释型</td>
    </tr>
    </table>

* **与其他语言的互操作性**

    <table>
    <tr>
        <td>Python</td>
        <td>C/C++</td>
        <td>Rust</td>
        <td>Go</td>
        <td>Zig</td>
        <td>Swift</td>
        <td>C#/.Net</td>
    </tr>
    </table>

* **支持的非 MVP 功能**

    <table>
    <tr>
        <td>多值</td>
        <td>大块内存操作</td>
        <td>符号扩展运算符</td>
        <td>非陷入式类型转换</td>
        <td>Name Section</td>
    </tr>
    </table>

* **支持的宿主 API**

    <table>
    <tr>
        <td>WASI</td>
        <td>自定义</td>
    </tr>
    </table>

* **非 Web 标准**

    <table>
    <tr>
        <td>WASI</td>
        <td>Gas Metering</td>
    </tr>
    </table>

* **使用者**

    - [wasmcloud](https://wasmcloud.dev/) - 编写可移植业务逻辑的平台
    - [Shareup](https://shareup.app/) - 为每个人提供快速、私密的共享
    - [WowCube](https://wowcube.com/) - 创新的游戏控制器和游戏平台
    - [txiki.js](https://github.com/saghul/txiki.js) - 一个小而强大的 JavaScript 运行时区

* **支持的平台**

    <table>
    <tr>
        <td>Windows</td>
        <td>Linux<br/>(任何架构)</td>
        <td>macOS<br/>(任何架构)</td>
        <td>FreeBSD<br/>(任何架构)</td>
        <td>Android</td><td>OpenWRT</td>
        <td>SBC/MCU</td>
        <td>Arduino</td>
    </tr>
    </table>



## <a name="motor"></a>[Motor](https://github.com/penberg/motor) <sup>[top⇈](#目录)</sup>
> Motor 是一个 WebAssembly 运行时，旨在安全高效地执行 WebAssembly 程序

* **编写语言**

    <table>
    <tr>
        <td>Rust</td>
    </tr>
    </table>

* **编译器框架**

    <table>
    <tr>
        <td>Dynasm</td>
    </tr>
    </table>


* **编译 / 执行模式**

    <table>
    <tr>
        <td>JIT</td>
    </tr>
    </table>

* **与其他语言的互操作性**

    - `N/A`

* **支持的 MVP 之外的特性**

    - `N/A`

* **支持的主机 API**

    - `N/A`

* **非网络标准**

    - `N/A`

* **使用者**

    - `N/A`

* **支持的平台**

    <table>
    <tr>
        <td>Linux</td>
        <td>macOS</td>
    </tr>
    </table>


## <a name="py-wasm"></a>[py-wasm](https://github.com/ethereum/py-wasm) <sup>[top⇈](#目录)</sup>
> WebAssembly 解释器的 Python 实现

* **编写语言**

    <table>
    <tr>
        <td>Python</td>
    </tr>
    </table>

* **编译器框架**

    <table>
    <tr>
        <td>定制化</td>
    </tr>
    </table>


* **编译 / 执行模式**

    <table>
    <tr>
        <td>解释执行</td>
    </tr>
    </table>

* **与其他语言的互操作性**

    - `N/A`

* **支持的 MVP 之外的特性**

    - `N/A`

* **支持的主机 API**

    - `N/A`

* **非网络标准**

    - `N/A`

* **使用者**

    - `N/A`

* **支持的平台**

    <table>
    <tr>
        <td>Linux</td>
        <td>macOS</td>
        <td>Windows</td>
    </tr>
    </table>


## <a name="serverless-wasm"></a>[Serverless wasm](https://github.com/Geal/serverless-wasm) <sup>[top⇈](#目录)</sup>
> 展示 WebAssembly 的潜力，展现其超越浏览器的能力

* **编写语言**

    <table>
    <tr>
        <td>Rust</td>
    </tr>
    </table>

* **编译器框架**

    <table>
    <tr>
        <td>N/A</td>
    </tr>
    </table>


* **编译 / 执行模式**

    <table>
    <tr>
        <td>N/A</td>
    </tr>
    </table>

* **与其他语言的互操作性**

    - `N/A`

* **支持的 MVP 之外的特性**

    - `N/A`

* **支持的主机 API**

    - `N/A`

* **非网络标准**

    - `N/A`

* **使用者**

    - `N/A`

* **支持的平台**

    <table>
    <tr>
        <td>Linux</td>
        <td>macOS</td>
    </tr>
    </table><table>
    <tr>
        <td>Wasmi</td>
        <td>Cranelift</td>
    </tr>
    </table>


* **Compilation / Execution modes**

    <table>
    <tr>
        <td>解释</td>
        <td>JIT</td>
    </tr>
    </table>

* **与其他语言的互操作性**

    - `N/A`

* **支持的非MVP特性**

    - `N/A`

* **支持的宿主API**

    - `N/A`

* **非Web标准**

    - `N/A`

* **使用者**

    - `N/A`

* **支持的平台**

    <table>
    <tr>
        <td>Linux</td>
        <td>macOS</td>
        <td>Windows</td>
    </tr>
    </table>


## <a name="swam"></a>[Swam](https://github.com/satabin/swam) <sup>[置顶⇈](#contents)</sup>
> Scala编写的WebAssembly引擎

* **使用的编程语言**

    <table>
    <tr>
        <td>Scala</td>
    </tr>
    </table>

* **编译器框架**

    <table>
    <tr>
        <td>自定义</td>
    </tr>
    </table>


* **编译 / 执行模式**

    <table>
    <tr>
        <td>解释</td>
    </tr>
    </table>

* **与其他语言的互操作性**

    - `N/A`

* **支持的非MVP特性**

    - `N/A`

* **支持的宿主API**

    - `N/A`

* **非Web标准**

    - `N/A`

* **使用者**

    - `N/A`

* **支持的平台**

    <table>
    <tr>
        <td>Linux</td>
        <td>macOS</td>
        <td>Windows</td>
    </tr>
    </table>


## <a name="vmir"></a>[VMIR](https://github.com/andoma/vmir) <sup>[置顶⇈](#contents)</sup>
> VMIR是一个独立的C库，可以解析和执行WebAssembly和LLVM位代码文件。

* **使用的编程语言**

    <table>
    <tr>
        <td>C</td>
    </tr>
    </table>

* **编译器框架**

    <table>
    <tr>
        <td>LLVM</td>
    </tr>
    </table>


* **编译 / 执行模式**

    <table>
    <tr>
        <td>解释</td>
        <td>JIT</td>
    </tr>
    </table>

* **与其他语言的互操作性**

    - `N/A`

* **支持的非MVP特性**

    - `N/A`

* **支持的宿主API**

    - `N/A`

* **非Web标准**"- `N/A`

* **使用者**

    - `N/A`
    - `N/A`

* **受支持的平台**

    <table>
    <tr>
        <td>Linux</td>
        <td>macOS</td>
        <td>Windows</td>
    </tr>
    </table>


## <a name="wac"></a>[wac](https://github.com/kanaka/wac) <sup>[top⇈](#contents)</sup>
> 用 C 语言编写的极简 WebAssembly 解释器。

* **编写语言**

    <table>
    <tr>
        <td>C 语言</td>
    </tr>
    </table>

* **编译器框架**

    <table>
    <tr>
        <td>自定义</td>
    </tr>
    </table>


* **编译/执行模式**

    <table>
    <tr>
        <td>解释执行</td>
    </tr>
    </table>

* **与其他语言的互操作性**

    - `N/A`

* **支持的非核心 MVP 特性**

    - `N/A`

* **支持的主机 API**

    - `N/A`

* **非 Web 标准**

    - `N/A`

* **使用者**

    - `N/A`

* **受支持的平台**

    <table>
    <tr>
        <td>Linux</td>
        <td>macOS</td>
        <td>Windows</td>
    </tr>
    </table>


## <a name="wagon"></a>[wagon](https://github.com/go-interpreter/wagon) <sup>[top⇈](#contents)</sup>
> 基于 Go 的 WebAssembly 解释器。

* **编写语言**

    <table>
    <tr>
        <td>Go 语言</td>
    </tr>
    </table>

* **编译器框架**

    <table>
    <tr>
        <td>自定义</td>
    </tr>
    </table>


* **编译/执行模式**

    <table>
    <tr>
        <td>解释执行</td>
    </tr>
    </table>

* **与其他语言的互操作性**

    - `N/A`

* **支持的非核心 MVP 特性**

    - `N/A`

* **支持的主机 API**

    - `N/A`

* **非 Web 标准**

    - `N/A`

* **使用者**

    - `N/A`

* **受支持的平台**

    <table>
    <tr>
        <td>Linux</td>
        <td>macOS</td>
        <td>Windows</td>
    </tr>
    </table>


## <a name="wakit"></a>[WAKit](https://github.com/akkyie/WAKit) <sup>[top⇈](#contents)</sup>
> 用 Swift 语言编写的 WebAssembly 运行时。

* **编写语言**

    <table>
    <tr>
        <td>Swift 语言</td>
    </tr>
    </table>"* **编译器框架**

    <table>
    <tr>
        <td>自定义</td>
    </tr>
    </table>


* **编译/执行模式**

    <table>
    <tr>
        <td>解释器</td>
    </tr>
    </table>

* **与其他语言互通性**

    - `N/A`

* **支持的非 MVP 特性**

    - `N/A`

* **支持的 Host API**

    - `N/A`

* **非 Web 标准**

    - `N/A`

* **被使用**

    - `N/A`

* **支持的平台**

    <table>
    <tr>
        <td>Linux</td>
        <td>macOS</td>
    </tr>
    </table>

## <a name="warpy"></a>[Warpy](https://github.com/kanaka/warpy) <sup>[top⇈](#内容)</sup>
> RPython 实现的 WebAssembly 解释器。

* **使用的语言**

    <table>
    <tr>
        <td>RPython</td>
    </tr>
    </table>

* **编译器框架**

    <table>
    <tr>
        <td>RPython</td>
    </tr>
    </table>


* **编译/执行模式**

    <table>
    <tr>
        <td>JIT</td>
    </tr>
    </table>

* **与其他语言互通性**

    - `N/A`

* **支持的非 MVP 特性**

    - `N/A`

* **支持的 Host API**

    - `N/A`

* **非 Web 标准**

    - `N/A`

* **被使用**

    - `N/A`

* **支持的平台**

    <table>
    <tr>
        <td>Linux</td>
        <td>macOS</td>
        <td>Windows</td>
    </tr>
    </table>


## <a name="wasmedge"></a>[WasmEdge](https://github.com/WasmEdge/WasmEdge) <sup>[top⇈](#内容)</sup>
> 一个轻量级、高性能、可扩展的 WebAssembly 运行时，用于云原生、边缘和分布式应用。属于 CNCF 项目。

* **使用的语言**

    <table>
    <tr>
        <td>C++</td>
    </tr>
    </table>

* **编译器框架**

    <table>
    <tr>
        <td>LLVM</td>
    </tr>
    </table>


* **编译/执行模式**

    <table>
    <tr>
        <td>解释器</td>
        <td>AOT</td>
    </tr>
    </table>

* **与其他语言互通性**

   <table>
    <tr>
        <td>Solidity</td>
        <td>Rust</td>
        <td>C/C++</td><td>Go/TinyGo</td>
        <td>JavaScript</td>
        <td>Python</td>
        <td>Grain</td>
        <td>Swift</td>
        <td>Zig</td>
        <td>Ruby</td>
    </tr>
    </table>

* **非MVP特性支持**

    <table>
    <tr>
        <td>批量内存操作</td>
        <td>SIMD</td>
        <td>多内存</td>
        <td>多值</td>
        <td>引用类型</td>
        <td>符号扩展指令</td>
        <td>非陷阱浮点数到整数转换</td>
        <td>可变全局变量</td>
    </tr>
    </table>

* **宿主API支持**

    <table>
    <tr>
        <td>WASI</td>
        <td>网络套接字</td>
        <td>TensorFlow</td>
        <td>命令接口</td>
        <td>图像处理</td>
    </tr>
    </table>

* **非web标准**

   <table>
    <tr>
        <td>WASI</td>
        <td>气体计量</td>
        <td>Ethereum环境接口</td>
        <td>Oasis</td>
        <td>Substrate</td>
    </tr>
    </table>

* **被以下项目使用**

    - [Suborbital](https://blog.suborbital.dev/suborbital-wasmedge)
    - [crun](https://github.com/containers/crun/pull/774)
    - [SuperEdge](https://github.com/superedge/superedge/pull/335)
    - [OpenYurt](https://github.com/openyurtio/openyurt.io/pull/85)
    - [Dapr](https://www.infoq.com/articles/webassembly-dapr-wasmedge/)
    - [Yomo](https://github.com/yomorun/yomo-wasmedge-tensorflow)
    - [Oasis ETH ParaTime](https://medium.com/oasis-protocol-project/the-oasis-eth-paratime-is-live-on-mainnet-33d8713ec870)

* **支持的平台**

    <table>
    <tr>
        <td>Linux</td>
        <td>macOS</td>
        <td>Windows</td>
        <td>Android</td>
        <td>OpenHarmony</td>
        <td>seL4 RTOS</td>
    </tr>
    </table>

## <a name="wasmer"></a>[Wasmer](https://github.com/wasmerio/wasmer) <sup>[返回顶部⇈](#content)</sup>
> Wasmer是一个独立的WebAssembly运行时，支持在浏览器外运行WebAssembly，支持WASI和Emscripten。

* **编写的语言**

    <table>
    <tr><td> Rust </td>
        <td> C++ </td>
    </tr>
    </table>

* ** 编译器框架 **

    <table>
    <tr>
        <td> Singlepass </td>
        <td> Cranelift（默认） </td>
        <td> LLVM </td>
    </tr>
    </table>


* ** 编译/执行模式 **

    <table>
    <tr>
        <td> JIT </td>
        <td> AOT </td>
    </tr>
    </table>

* ** 与其他语言的互操作性 **

    <table>
    <tr>
        <td> Rust </td>
        <td> PHP </td>
        <td> C </td>
        <td> C ++ </td>
        <td> Python </td>
        <td> Go </td>
        <td> PHP </td>
        <td> Java </td>
        <td> Ruby </td>
        <td> Postgres </td>
        <td> C#/.Net </td>
        <td> Elixir </td>
        <td> R </td>
        <td> D </td>
        <td> Swift </td>
    </tr>
    </table>

* ** 非MVP支持的功能 **

    <table>
    <tr>
        <td> Threads </td>
        <td> SIMD </td>
        <td> Multi Value返回 </td>
        <td>名称部分 </td>
        <td>批量内存操作 </td>
        <td>符号扩展指令 </td>
    </tr>
    </table>

* ** 主机API支持 **

    <table>
    <tr>
        <td> Emscripten </td>
        <td> WASI </td>
    </tr>
    </table>

* ** 非Web标准 **

    <table>
    <tr>
        <td> WASI </td>
        <td> wasm-c-api </td>
    </tr>
    </table>

* **使用者**

    - [ Spacemesh Virtual Machine ]（https://github.com/spacemeshos/svm） - Spacemesh智能合约vm
    - [ CosmWasm ]（https://github.com/CosmWasm/cosmwasm） - 用于运行wasm智能合约的兼容Cosmos的库
    - [ NEAR Protocol ]（https://github.com/near/nearcore） - NEAR协议的参考客户端
    - [ Dprint ]（https://github.com/dprint/dprint） - 插件式和可配置的代码格式化平台

* **支持的平台**

    <table>
    <tr>
        <td> Linux（x64，aarch64，x86） </td>
        <td> macOS（x64，arm64） </td>
        <td> Windows（x64，x86） </td>
        <td> FreeBSD（x64） </td>
        <td> Android </td>
    </tr>
    </table>## <a name="wasmi"></a>[Wasmi](https://github.com/paritytech/wasmi) <sup>[返回顶部⇈](#contents)</sup>
> 一个Wasm解释器。

* **编写语言**

    <table>
    <tr>
        <td>Rust</td>
    </tr>
    </table>

* **编译器框架**

    <table>
    <tr>
        <td>自定义</td>
    </tr>
    </table>


* **编译/执行模式**

    <table>
    <tr>
        <td>解释执行</td>
    </tr>
    </table>

* **与其他语言的兼容性**

    - `N/A`

* **不是MVP功能的支持**

    - `N/A`

* **主机API的支持**

    - `N/A`

* **非Web标准**

    - `N/A`

* **被谁使用**

    - `N/A`

* **支持的平台**

    <table>
    <tr>
        <td>Linux</td>
        <td>macOS</td>
        <td>Windows</td>
    </tr>
    </table>

## <a name="wasmo"></a>[Wasmo](https://github.com/appcypher/wasmo) <sup>[返回顶部⇈](#contents)</sup>
> 基于LLVM的WebAssembly运行时和编译器。

* **编写语言**

    <table>
    <tr>
        <td>Rust</td>
    </tr>
    </table>

* **编译器框架**

    <table>
    <tr>
        <td>LLVM</td>
    </tr>
    </table>


* **编译/执行模式**

    <table>
    <tr>
        <td>JIT</td>
    </tr>
    </table>

* **与其他语言的兼容性**

    - `N/A`

* **不是MVP功能的支持**

    - `N/A`

* **主机API的支持**

    - `N/A`

* **非Web标准**

    - `N/A`

* **被谁使用**

    - `N/A`

* **支持的平台**

    <table>
    <tr>
        <td>macOS</td>
    </tr>
    </table>

## <a name="wasmrt"></a>[wasmrt](https://github.com/rhitchcock/wasmrt) <sup>[返回顶部⇈](#contents)</sup>
> wasmrt是一个用于本地执行WebAssembly模块（一开始是虚拟化的，最终是JIT）的运行时。

* **编写语言**

    <table>
    <tr>
        <td>C++</td>
    </tr>
    </table>

* **编译器框架**

    <table>
    <tr>
        <td>自定义</td>
    </tr>
    </table>


* **编译/执行模式**

    <table>
    <tr>
        <td>解释执行</td>
    </tr>
    </table>* **与其他语言的互通性**

    - `N/A`

* **支持的非 MVP 功能**

    - `N/A`

* **支持的主机 API**

    - `N/A`

* **非 Web 标准**

    - `N/A`

* **使用者**

    - `N/A`

* **支持的平台**

    <table>
    <tr>
        <td>Linux</td>
        <td>macOS</td>
        <td>Windows</td>
    </tr>
    </table>

## <a name="wasmtime"></a>[Wasmtime](https://github.com/CraneStation/wasmtime) <sup>[top⇈](#contents)</sup>
> Wasmtime 是一個獨立的只支援 WebAssembly的運行時，使用了 Cranelift 的虛擬碼編譯器（JIT）

* **用于编写的语言**

    <table>
    <tr>
        <td>Rust</td>
    </tr>
    </table>

* **编译器框架**

    <table>
    <tr>
        <td>Cranelift</td>
    </tr>
    </table>


* **编译 / 执行模式**

    <table>
    <tr>
        <td>JIT</td>
    </tr>
    </table>

* **与其他语言的互通性**

    <table>
    <tr>
        <td>Python</td>
    </tr>
    </table>

* **支持的非 MVP 功能**

    <table>
    <tr>
        <td>接口类型</td>
    </tr>
    </table>

* **支持的主机 API**

    <table>
    <tr>
        <td>WASI</td>
    </tr>
    </table>

* **非 Web 标准**

    <table>
    <tr>
        <td>WASI</td>
        <td>wasm-c-api</td>
    </tr>
    </table>

* **使用者**

    - `N/A`

* **支持的平台**

    <table>
    <tr>
        <td>Linux</td>
        <td>macOS</td>
        <td>Windows</td>
    </tr>
    </table>

## <a name="wasmvm1"></a>[WasmVM](https://github.com/WasmVM/WasmVM) <sup>[top⇈](#contents)</sup>
> 一個非官方的獨立 WebAssembly 處理器虛擬機

* **用于编写的语言**

    <table>
    <tr>
        <td>C++</td>
    </tr>
    </table>

* **编译器框架**

    <table>
    <tr>
        <td>自定义</td>
    </tr>
    </table>


* **编译 / 执行模式**

    <table>
    <tr>
        <td>解释</td>
    </tr>
    </table>

* **与其他语言的互通性**

    - `N/A`

* **支持的非 MVP 功能**

    - `N/A`* **支持的主机API**

    - `N/A`

* **非 Web 标准**

    - `N/A`

* **被使用于**

    - `N/A`

* **支持的平台**

    <table>
    <tr>
        <td>Linux</td>
        <td>macOS</td>
        <td>Windows</td>
    </tr>
    </table>

## <a name="wasmvm2"></a>[wasmvm](https://github.com/kogai/wasvm) <sup>[返回页首⇈](#contents)</sup>
> WebAssembly 虚拟机，旨在适用于嵌入式系统。

* **编写的编程语言**

    <table>
    <tr>
        <td>Rust</td>
    </tr>
    </table>

* **编译器框架**

    <table>
    <tr>
        <td>Life</td>
    </tr>
    </table>


* **编译 / 执行模式**

    <table>
    <tr>
        <td>解释</td>
    </tr>
    </table>

* **与其他语言的互操作性**

    - `N/A`

* **非 MVP 功能支持**

    - `N/A`

* **支持的主机API**

    - `N/A`

* **非 Web 标准**

    - `N/A`

* **被使用于**

    - `N/A`

* **支持的平台**

    <table>
    <tr>
        <td>Linux</td>
        <td>macOS</td>
        <td>Windows</td>
    </tr>
    </table>

## <a name="wavm"></a>[WAVM](https://github.com/WAVM/WAVM) <sup>[返回页首⇈](#contents)</sup>
> WebAssembly 虚拟机，旨在适用于嵌入式系统。

* **编写的编程语言**

    <table>
    <tr>
        <td>C++</td>
        <td>Python</td>
    </tr>
    </table>

* **编译器框架**

    <table>
    <tr>
        <td>LLVM</td>
    </tr>
    </table>


* **编译 / 执行模式**

    <table>
    <tr>
        <td>JIT</td>
    </tr>
    </table>

* **与其他语言的互操作性**

    - `N/A`

* **非 MVP 功能支持**

    <table>
    <tr>
        <td>线程</td>
        <td>SIMD</td>
        <td>多结果和块参数</td>
        <td>异常处理</td>
        <td>Name Section</td>
        <td>Reference Types</td>
        <td>Bulk Memory Operations</td>
        <td>Sign Extension Instructions</td>
    </tr>
    </table>

* **支持的主机API**

    <table>
    <tr>
        <td>WASI</td><td>Emscripten</td>
        <td>WAVIX</td>
    </tr>
    </table>

* **非 Web 标准**

    - [x] WASI

* **被使用**

    - `N/A`

* **支持平台**

    <table>
    <tr>
        <td>Linux</td>
        <td>macOS</td>
        <td>Windows</td>
    </tr>
    </table>

## <a name="webassembly"></a>[WebAssembly](https://github.com/dcodeIO/webassembly) <sup>[返回⇈](#目录)</sup>
> 开发 WebAssembly 模块的实验性工具和运行时的精简版本

* **编写语言**

    <table>
    <tr>
        <td>JavaScript</td>
    </tr>
    </table>

* **编译器框架**

    <table>
    <tr>
        <td>NodeJS(V8)</td>
    </tr>
    </table>


* **编译/执行模式**

    <table>
    <tr>
        <td>JIT</td>
    </tr>
    </table>

* **与其他编程语言的互操作性**

    - `N/A`

* **支持的非 MVP 特性**

    - `N/A`

* **支持的主机 API**

    - `N/A`

* **非 Web 标准**

    - `N/A`

* **被使用**

    - `N/A`

* **支持平台**

   <table>
    <tr>
        <td>Linux</td>
        <td>macOS</td>
        <td>Windows</td>
    </tr>
    </table>


## <a name="wamr"></a>[WAMR](https://github.com/bytecodealliance/wasm-micro-runtime) <sup>[返回⇈](#目录)</sup>
> WebAssembly 微运行时（WAMR）是一个占用空间很小的独立 WebAssembly（WASM）运行时

* **编写语言**

    <table>
    <tr>
        <td>C</td>
    </tr>
    </table>


* **编译器框架**

    <table>
    <tr>
        <td>Custom</td>
    </tr>
    </table>


* **编译/执行模式**

    <table>
    <tr>
        <td>解释型</td>
        <td>AOT 模块化</td>
        <td>JIT</td>
    </tr>
    </table>

* **与其他编程语言的互操作性**

    - `N/A`

* **支持的非 MVP 特性**

    - [非陷阱浮点数向整数转换](https://github.com/WebAssembly/nontrapping-float-to-int-conversions)
    - [符号扩展运算符](https://github.com/WebAssembly/sign-extension-ops)- [批量内存操作](https://github.com/WebAssembly/bulk-memory-operations)
    - [共享内存](https://github.com/WebAssembly/threads/blob/main/proposals/threads/Overview.md#shared-linear-memory)
    - [多值](https://github.com/WebAssembly/multi-value)
    - [尾调用](https://github.com/WebAssembly/tail-call)


* **支持的主机 API**

    - [wasm-c-api](https://github.com/WebAssembly/wasm-c-api)

* **非 Web 标准**

    - [x] WASI

* **被使用的**

    - `N/A`

* **支持的平台**

    <table>
    <tr>
        <td>Linux</td>
        <td>macOS</td>
        <td>Windows</td>
        <td>AliOS-Things</td>
        <td>Android</td>
        <td>Linux-SGX</td>
        <td>Nuttx</td>
        <td>VxWorks</td>
        <td>Zephyr</td>
    </tr>
    </table>



## <a name="twvm"></a>[TWVM](https://github.com/Becavalier/TWVM) <sup>[top⇈](#contents)</sup>
> 一个小巧高效的 WebAssembly 虚拟机。

* **编写语言**

    <table>
    <tr>
        <td>C++</td>
    </tr>
    </table>

* **编译器框架**

    <table>
    <tr>
        <td>自定义</td>
    </tr>
    </table>


* **编译 / 运行模式**

    <table>
    <tr>
        <td>解释执行</td>
    </tr>
    </table>

* **与其他语言的互操作性**

    - `N/A`

* **支持的非 MVP 特性**

    - `N/A`

* **支持的主机 API**

    - `N/A`

* **非 Web 标准**

    - `N/A`

* **被使用的**

    - `N/A`

* **支持的平台**

    <table>
    <tr>
        <td>Linux</td>
        <td>macOS</td>
        <td>Windows</td>
    </tr>
    </table>



## <a name="wazero"></a>[wazero](https://wazero.io) <sup>[top⇈](#contents)</sup>
> wazero 是一个遵循 WebAssembly 1.0 和 2.0 规范的运行时，使用 Go 编写，零平台依赖。

* **编写语言**

    <table>
    <tr>
        <td>Go</td>
    </tr>
    </table>

* **编译器框架**

    <table>
    <tr>
        <td>自定义</td>
    </tr>
    </table>


* **编译 / 运行模式**

    <table>
    <tr><table>
        <td>解释型</td>
        <td>JIT</td>
    </tr>
    </table>

* **与其他语言的互操作性**

    - `不适用`

* **支持的非MVP功能**

    <table>
    <tr>
        <td>批量内存操作</td>
        <td>可变全局导入/导出</td>
        <td>符号扩展运算符</td>
        <td>多返回值</td>
        <td>名称部分</td>
        <td>非陷阱型浮点数到整数转换</td>
        <td>引用类型</td>
        <td>SIMD</td>
    </tr>
    </table>

* **支持的主机API**

    <table>
    <tr>
        <td>AssemblyScript</td>
        <td>WASI</td>
    </tr>
    </table>

* **非网络标准**

    <table>
    <tr>
        <td>WASI</td>
    </tr>
    </table>

* **被以下项目使用**

    - [dapr](https://github.com/dapr/dapr) - 简化微服务连接的API
    - [trivy](https://github.com/aquasecurity/trivy) - 容器和其他工件的漏洞/配置错误/秘密扫描器
    - [wapc-go](https://github.com/wapc/wapc-go) - waPC兼容模块的WebAssembly宿主运行时

* **支持的平台**

    <table>
    <tr>
        <td>FreeBSD (amd64)</td>
        <td>Linux (amd64、arm64、riscv64)</td>
        <td>macOS (amd64)</td>
        <td>Windows (amd64)</td>
    </tr>
    </table>
-------------------

## 许可证

[![CC0](http://mirrors.creativecommons.org/presskit/buttons/88x31/svg/cc-zero.svg)](https://creativecommons.org/publicdomain/zero/1.0/)

在法律许可的范围内，[Steve Akinyemi](https://github.com/appcypher)已放弃本作品的所有版权和相关或附属权利。