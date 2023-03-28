# 精选 WebAssembly 运行时 ![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)

本存储库包含一个虚拟机和工具的列表，该列表执行WebAssembly（wasm）格式并/或将其编译为可执行机器代码 :octocat:

#### 图例
:rocket: - 活跃开发</br>
:sleeping: - 未维护（距上次提交已超过一年）</br>

## 目录

- [aWsm](#awsm) <sup><sup>:rocket:</sup></sup></br>
- [Wasm CMM](#cmm) <sup><sup>:sleeping:</sup></sup></br>
- [EOSVM](#eosvm) <sup><sup>:rocket:</sup></sup></br>
- [FDVM](#fdvm) <sup><sup>:sleeping:</sup></sup></br>
- [Fizzy](#fizzy) <sup><sup>:rocket:</sup></sup></br>
- [GraalWASM](#graalwasm) <sup><sup>:rocket:</sup></sup></br>
- [新月和报告](#happy-new-moon-with-report) <sup><sup>:rocket:</sup></sup></br>
- [inNative](#innative) <sup><sup>:rocket:</sup></sup></br>
- [生命](#life) <sup><sup>:sleeping:</sup></sup></br>
- [Lucet](#lucet) <sup><sup>:rocket:</sup></sup></br>
- [wasm3](#wasm3) <sup><sup>:rocket:</sup></sup></br>
- [Motor](#motor) <sup><sup>:sleeping:</sup></sup></br>
- [py-wasm](#py-wasm) <sup><sup>:sleeping:</sup></sup></br>
- [无服务器Wasm](#serverless-wasm) <sup><sup>:sleeping:</sup></sup></br>
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


## <a name="awsm"></a>[aWsm](https://github.com/gwsystems/aWsm) <sup>[回到页首⇈](#contents)</sup>

> aWsm 是一个将 WebAssembly (Wasm) 代码编译成 LLVM 字节码，再编译成在各种平台上可以运行的沙箱二进制代码的编译器和运行时。

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


* **编译 / 执行模式**

    <table>
    <tr>
        <td>AOT</td>
    </tr>
    </table>

* **与其他语言互操作性**

    <table>
    <tr>
        <td>C</td>
    </tr>
    </table>

* **支持的非 MVP 特性**

    - `N/A`

* **支持的主机 API**

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
    </tr>
    </table>


## <a name="cmm"></a>[Wasm 的 CMM](https://github.com/SimonJF/cmm_of_wasm) <sup>[回到页首⇈](#contents)</sup>
> 一个将 WebAssembly 编译至本地代码的编译器，采用 OCaml 后端。

* **编写语言**

    <table>
    <tr>
        <td>OCaml</td>
    </tr>
    </table>

* **编译器框架**

    <table>
    <tr>
        <td>OCaml 编译器</td>
    </tr>
    </table>


* **编译 / 执行模式**

    <table>
    <tr>
        <td>AOT</td>
    </tr>
    </table>

* **与其他语言互操作性**

    - `N/A`

* **支持的非 MVP 特性**

    - `N/A`

* **支持的主机 API**

    - `N/A`

* **非 Web 标准**

    - `N/A`

* **被使用于**

    - `N/A`

* **支持的平台**

    <table><table>
        <td>Linux</td>
        <td>macOS</td>
    </tr>
    </table>

## <a name="eosvm"></a>[EOSVM](https://github.com/EOSIO/eos-vm) <sup>[返回顶部⇈](#contents)</sup>
> EOS VM是从WebAssembly引擎需要的Web浏览器或标准开发设计的要求远远超出的区块链应用所设计的。

* **使用的语言**

    <table>
    <tr>
        <td>C++</td>
    </tr>
    </table>

* **编译器框架**

    <table>
    <tr>
        <td>定制版</td>
    </tr>
    </table>


* **编译与执行模式**

    <table>
    <tr>
        <td>解释执行</td>
    </tr>
    </table>

* **与其他语言的互操作性**

    - `N/A`

* **支持的非-MVP功能**

    - `N/A`

* **支持的 Host APIs**

    - `N/A`

* **支持的非 web 标准**

    - `N/A`

* **被支持**

    - `N/A`

* **支持的平台**

    <table>
    <tr>
        <td>Linux</td>
        <td>macOS</td>
        <td>Windows</td>
    </tr>
    </table>


## <a name="fdvm"></a>[FDVM](https://github.com/funcdef/fdvm) <sup>[返回顶部⇈](#contents)</sup>
> 用于开发服务器端 WebAssembly 应用程序的 WASM 运行时。

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


* **编译与执行模式**

    <table>
    <tr>
        <td>JIT</td>
    </tr>
    </table>

* **与其他语言的互操作性**

    - `N/A`

* **支持的非-MVP功能**

    - `N/A`

* **支持的 Host APIs**

    - `N/A`

* **支持的非 web 标准**

    - `N/A`

* **被支持**

    - `N/A`

* **支持的平台**

    <table>
    <tr>
        <td>Linux</td>
        <td>macOS</td>
        <td>Windows</td>
    </tr>
    </table>


## <a name="fizzy"></a>[Fizzy](https://github.com/wasmx/fizzy) <sup>[返回顶部⇈](#contents)</sup>
> Fizzy旨在成为一个快速，确定性和严谨的WebAssembly解释器，使用C++编写。* **编写语言**

    <table>
    <tr>
        <td>C++</td>
    </tr>
    </table>

* **编译器框架**

    - `N/A`

* **编译/执行模式**

    <table>
    <tr>
        <td>解释器</td>
    </tr>
    </table>

* **与其他语言的互操作性**

    - `N/A`

* **支持的非 MVP 功能**

    - `N/A`

* **支持的主机 API**

    <table>
    <tr>
        <td>WASI</td>
    </tr>
    </table>

* **非 web 标准**

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
GraalWasm 是在 GraalVM 中实现的 WebAssembly 引擎。它可以解释和编译二进制格式的 WebAssembly 程序，或者嵌入到其他程序中。

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


* **编译/执行模式**

    <table>
    <tr>
        <td>JIT</td>
    </tr>
    </table>

* **与其他语言的互操作性**

    <table>
    <tr>
        <td>Java</td>
        <td>JVM</td>
        <td>GraalVM 支持的语言</td>
    </tr>
    </table>

* **支持的非 MVP 功能**

    <table>
    <tr>
    </tr>
    </table>

* **支持的主机 API**

    <table>
    <tr>
        <td>WASI</td>
    </tr>
    </table>

* **非 web 标准**

    <table>
    <tr>
        <td>WASI</td>
    </tr>
    </table>

* **使用者**

    - [GraalVM JavaScript](https://github.com/graalvm/graaljs) - JavaScript 编程语言的高性能实现。

* **支持的平台**

    <table>
    <tr>
        <td>Linux</td>
        <td>macOS</td>
        <td>Windows</td>
    </tr>
    </table>## <a name="happy-new-moon-with-report"></a>[快乐的新月与报告](https://github.com/fishjd/HappyNewMoonWithReport) <sup>[top⇈](#contents)</sup>

快乐的新月与报告是一个完全用 Java 编写的开源的 WebAssembly 实现。它用于在 Java 中运行或测试 WebAssembly 模块 (*.wasm) 。

* **编写语言**

    <table>
    <tr>
        <td>Java</td>
    </tr>
    </table>

* **编译器框架**

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
        <td>JVM 语言</td>
    </tr>
    </table>

* **支持的非 MVP 特性**

    <table>
    <tr>
    	<td>符号扩展</td>
    </tr>
    </table>

* **支持的主机 API**

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

## <a name="innative"></a>[inNative](https://github.com/innative-sdk/innative) <sup>[top⇈](#contents)</sup>
> 一种 WebAssembly 的 AOT（ahead-of-time）编译器，可创建 C 兼容的二进制文件，可以作为沙盒化插件进行动态加载，也可以作为直接与操作系统交互的独立可执行文件。

* **编写语言**

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
        <td>AOT</td>
        <td>JIT</td>
    </tr>
    </table>

* **与其他语言的互操作性**

    - `N/A`

* **支持的非 MVP 特性**

    <table>
    <tr>
        <td>线程</td>
        <td>多个结果和块参数</td>
        <td>Name Section</td>
        <td>批量内存操作</td>
        <td>符号扩展指令</td>enschaften</tr>
    </table><td>非陷阱转换说明</td>
    </tr>
    </table>

* **支持的主机API**

    <table>
    <tr>
        <td>定制</td>
    </tr>
    </table>

* **非Web标准**

    - `N/A`

* **使用者**

    - `N/A`

* **支持平台**

    <table>
    <tr>
        <td>Linux</td>
        <td>macOS</td>
        <td>Windows</td>
    </tr>
    </table>


## <a name="life"></a>[Life](https://github.com/perlin-network/life) <sup>[top⇈](#contents)</sup>
> Life是一个安全且快速的WebAssembly VM，由Perlin Network使用Go编写，专为去中心化应用程序而构建。

* **所使用的语言**

    <table>
    <tr>
        <td>Go</td>
    </tr>
    </table>

* **编译器框架**

    <table>
    <tr>
        <td>定制</td>
    </tr>
    </table>


* **编译/执行模式**

    <table>
    <tr>
        <td>解释型</td>
    </tr>
    </table>

* **与其他语言的互操作性**

    - `N/A`

* **支持的非MVP特性**

    - `N/A`

* **支持的主机API**

    - `N/A`

* **非Web标准**

    - `N/A`

* **使用者**

    - `N/A`

* **支持平台**

    <table>
    <tr>
        <td>Linux</td>
        <td>macOS</td>
        <td>Windows</td>
    </tr>
    </table>

## <a name="lucet"></a>[Lucet](https://github.com/fastly/lucet) <sup>[top⇈](#contents)</sup>
> Lucet是原生的WebAssembly编译器和运行时。 它旨在安全地在您的应用程序内执行不受信任的WebAssembly程序。

> 注意：Lucet现已进入维护模式。 工作已移至[wasmtime](https://github.com/bytecodealliance/wasmtime/)。

* **所使用的语言**

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

* **支持的主机API**<table>
    <tr>
        <td>WASI</td>
    </tr>
    </table>

* **非 web 标准**

    <table>
    <tr>
        <td>WASI</td>
    </tr>
    </table>

* **使用方**

    - `N/A`

* **支持的平台**

    <table>
    <tr>
        <td>Linux</td>
    </tr>
    </table>


## <a name="wasm3"></a>[Wasm3](https://github.com/wasm3/wasm3) <sup>[返回顶部⇈](#contents)</sup>
> 🚀  一个高性能的 WebAssembly 解释器

* **编写语言**

    <table>
    <tr>
        <td>C</td>
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
        <td>解释</td>
    </tr>
    </table>

* **与其他语言的互通性**

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

* **支持的非SSM功能**

    <table>
    <tr>
        <td>多值</td>
        <td>大块内存操作</td>
        <td>符号扩展操作符</td>
        <td>非陷阱转换</td>
        <td>名称段</td>
    </tr>
    </table>

* **支持的主机API**

    <table>
    <tr>
        <td>WASI</td>
        <td>自定义</td>
    </tr>
    </table>

* **非web标准**

    <table>
    <tr>
        <td>WASI</td>
        <td>燃气计量</td>
    </tr>
    </table>

* **使用方**

    - [wasmcloud](https://wasmcloud.dev/) - 可移植业务逻辑的平台
    - [Shareup](https://shareup.app/) - 快速、私密的共享工具
    - [WowCube](https://wowcube.com/) - 创新型游戏平台
    - [txiki.js](https://github.com/saghul/txiki.js) -一个轻量级且强大的JavaScript运行时

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



## <a name="motor"></a>[马达](https://github.com/penberg/motor) <sup>[返回顶部⇈](#contents)</sup>
> Motor是一个WebAssembly运行时，旨在安全高效地执行WebAssembly程序

* **使用的编程语言**

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


* **编译/执行模式**

    <table>
    <tr>
        <td>JIT</td>
    </tr>
    </table>

* **与其他编程语言的互操作性**

    - `N/A`

* **支持的非MVP特性**

    - `N/A`

* **支持的主机API**

    - `N/A`

* **非Web标准**

    - `N/A`

* **应用**

    - `N/A`

* **支持的平台**

    <table>
    <tr>
        <td>Linux</td>
        <td>macOS</td>
    </tr>
    </table>


## <a name="py-wasm"></a>[py-wasm](https://github.com/ethereum/py-wasm) <sup>[返回顶部⇈](#contents)</sup>
> WebAssembly模拟器的Python实现

* **使用的编程语言**

    <table>
    <tr>
        <td>Python</td>
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
        <td>解释型</td>
    </tr>
    </table>

* **与其他编程语言的互操作性**

    - `N/A`

* **支持的非MVP特性**

    - `N/A`

* **支持的主机API**

    - `N/A`

* **非Web标准**

    - `N/A`

* **应用**

    - `N/A`

* **支持的平台**

    <table>
    <tr>
        <td>Linux</td>
        <td>macOS</td>
        <td>Windows</td>
    </tr>
    </table>


## <a name="serverless-wasm"></a>[无服务器wasm](https://github.com/Geal/serverless-wasm) <sup>[返回顶部⇈](#contents)</sup>
> WebAssembly潜在的浏览器外的小演示

* **使用的编程语言**

    <table>
    <tr>
        <td>Rust</td>
    </tr>
    </table>

* **编译器框架**<table>
    <tr>
        <td>Wasmi</td>
        <td>Cranelift</td>
    </tr>
</table>


* **编译/执行模式**

    <table>
    <tr>
        <td>解释</td>
        <td>JIT</td>
    </tr>
    </table>

* **与其他语言的互操作性**

    - `不适用`

* **支持的非MVP特性**

    - `不适用`

* **支持的主机API**

    - `不适用`

* **非Web标准**

    - `不适用`

* **被使用**

    - `不适用`

* **支持的平台**

    <table>
    <tr>
        <td>Linux</td>
        <td>macOS</td>
        <td>Windows</td>
    </tr>
    </table>


## <a name="swam"></a>[Swam](https://github.com/satabin/swam) <sup>[返回⇈](#目录)</sup>
> 用Scala编写的WebAssembly引擎

* **写入的语言**

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


* **编译/执行模式**

    <table>
    <tr>
        <td>解释</td>
    </tr>
    </table>

* **与其他语言的互操作性**

    - `不适用`

* **支持的非MVP特性**

    - `不适用`

* **支持的主机API**

    - `不适用`

* **非Web标准**

    - `不适用`

* **被使用**

    - `不适用`

* **支持的平台**

    <table>
    <tr>
        <td>Linux</td>
        <td>macOS</td>
        <td>Windows</td>
    </tr>
    </table>


## <a name="vmir"></a>[VMIR](https://github.com/andoma/vmir) <sup>[返回⇈](#目录)</sup>
> VMIR是用C编写的一个独立库，可以解析和执行WebAssembly和LLVM Bitcode文件

* **写入的语言**

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


* **编译/执行模式**

    <table>
    <tr>
        <td>解释</td>
        <td>JIT</td>
    </tr>
    </table>

* **与其他语言的互操作性**

    - `不适用`

* **支持的非MVP特性**

    - `不适用`

* **支持的主机API**

    - `不适用`

* **非Web标准**- `N/A`

* **使用者**

    - `N/A`
    - `N/A`

* **支持的平台**

    <table>
    <tr>
        <td>Linux</td>
        <td>macOS</td>
        <td>Windows</td>
    </tr>
    </table>


## <a name="wac"></a>[wac](https://github.com/kanaka/wac) <sup>[返回⇈](#contents)</sup>
> 一个用C编写的WebAssembly解释器。

* **编写的语言**

    <table>
    <tr>
        <td>C</td>
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

* **支持的非MVP功能**

    - `N/A`

* **主机API支持**

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


## <a name="wagon"></a>[wagon](https://github.com/go-interpreter/wagon) <sup>[返回⇈](#contents)</sup>
> Go语言的基于WebAssembly的解释器，为Go打造。

* **编写的语言**

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


* **编译/执行模式**

    <table>
    <tr>
        <td>解释执行</td>
    </tr>
    </table>

* **与其他语言的互操作性**

    - `N/A`

* **支持的非MVP功能**

    - `N/A`

* **主机API支持**

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


## <a name="wakit"></a>[WAKit](https://github.com/akkyie/WAKit) <sup>[返回⇈](#contents)</sup>
> 用Swift编写的WebAssembly运行时。

* **编写的语言**

    <table>
    <tr>
        <td>Swift</td>
    </tr>
    </table>* **编译器框架**

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

* **与其他语言互操作性**

     - `N/A`

* **支持的非-MVP特性**

     - `N/A`

* **支持的Host API**

     - `N/A`

* **非Web标准**

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

## <a name="warpy"></a>[Warpy](https://github.com/kanaka/warpy) <sup>[top⇈](#contents)</sup>
> RPython中的WebAssembly解释器。

* **编写的语言**

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

* **与其他语言互操作性**

     - `N/A`

* **支持的非-MVP特性**

     - `N/A`

* **支持的Host API**

     - `N/A`

* **非Web标准**

     - `N/A`

* **被以下项目使用**

     - `N/A`

* **支持的平台**

     <table>
     <tr>
         <td>Linux</td>
         <td>macOS</td>
         <td>Windows</td>
     </tr>
     </table>


## <a name="wasmedge"></a>[WasmEdge](https://github.com/WasmEdge/WasmEdge) <sup>[top⇈](#contents)</sup>
> 用于云原生、边缘和分布式应用的轻量级、高性能和可扩展的WebAssembly运行时。CNCF项目。

* **编写的语言**

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

* **与其他语言互操作性**

     <table>
     <tr>
         <td>Solidity</td>
         <td>Rust</td>
         <td>C/C++</td>
     </tr>
     </table><table>
    <tr>
        <td>Go/TinyGo</td>
        <td>JavaScript</td>
        <td>Python</td>
        <td>Grain</td>
        <td>Swift</td>
        <td>Zig</td>
        <td>Ruby</td>
    </tr>
    </table>

* **支持的非MVP特性**

    <table>
    <tr>
        <td>批量内存操作</td>
        <td>SIMD</td>
        <td>多内存</td>
        <td>多值</td>
        <td>引用类型</td>
        <td>符号扩展指令</td>
        <td>非陷阱浮点转整数</td>
        <td>可变全局</td>
    </tr>
    </table>

* **支持的主机API**

    <table>
    <tr>
        <td>WASI</td>
        <td>网络套接字</td>
        <td>TensorFlow</td>
        <td>命令接口</td>
        <td>图像处理</td>
    </tr>
    </table>

* **非Web标准**

   <table>
    <tr>
        <td>WASI</td>
        <td>燃气计量</td>
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

## <a name="wasmer"></a>[Wasmer](https://github.com/wasmerio/wasmer) <sup>[返回目录⇈](#contents)</sup>
> Wasmer是一个独立的WebAssembly运行时，用于在浏览器外运行WebAssembly，支持WASI和Emscripten。

* **编写的语言**

    <table>
    <tr><td> Rust </td>
        <td> C++ </td>
    </tr>
    </table>

* **编译器框架**

    <table>
    <tr>
        <td>Singlepass</td>
        <td>Cranelift (默认)</td>
        <td>LLVM</td>
    </tr>
    </table>


* **编译/执行模式**

    <table>
    <tr>
        <td>JIT</td>
        <td>AOT</td>
    </tr>
    </table>

* **与其他语言的互操作性**

    <table>
    <tr>
        <td>Rust</td>
        <td>PHP</td>
        <td>C</td>
        <td>C++</td>
        <td>Python</td>
        <td>Go</td>
        <td>PHP</td>
        <td>Java</td>
        <td>Ruby</td>
        <td>Postgres</td>
        <td>C#/.Net</td>
        <td>Elixir</td>
        <td>R</td>
        <td>D</td>
        <td>Swift</td>
    </tr>
    </table>

* **支持的非MVP功能**

    <table>
    <tr>
        <td>Threads</td>
        <td>SIMD</td>
        <td>多值返回</td>
        <td>Name Section</td>
        <td>Bulk Memory Operations</td>
        <td>Sign Extension Instructions</td>
    </tr>
    </table>

* **主机API支持**

    <table>
    <tr>
        <td>Emscripten</td>
        <td>WASI</td>
    </tr>
    </table>

* **非Web标准**

    <table>
    <tr>
        <td>WASI</td>
        <td>wasm-c-api</td>
    </tr>
    </table>

* **使用者**

    - [Spacemesh Virtual Machine](https://github.com/spacemeshos/svm) - Spacemesh智能合约虚拟机
    - [CosmWasm](https://github.com/CosmWasm/cosmwasm) - 用于运行wasm智能合约的与Cosmos兼容的库
    - [NEAR Protocol](https://github.com/near/nearcore) - NEAR协议的参考客户端
    - [Dprint](https://github.com/dprint/dprint) - 可插拔和可配置的代码格式化平台

* **支持的平台**

    <table>
    <tr>
        <td>Linux（x64，aarch64，x86）</td>
        <td>macOS（x64，arm64）</td>
        <td>Windows（x64，x86）</td>
        <td>FreeBSD（x64）</td>
        <td>Android</td>
    </tr>
    </table>"## <a name="wasmi"></a>[Wasmi](https://github.com/paritytech/wasmi) <sup>[返回顶部⇈](#目录)</sup>
> 一个 Wasm 解释器.

* **使用语言**

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


* **编译 / 执行模式**

    <table>
    <tr>
        <td>解释型</td>
    </tr>
    </table>

* **与其他语言的互操作性**

    - `N/A`

* **支持的非 MVP 功能**

    - `N/A`

* **支持的主机API**

    - `N/A`

* **非 Web 标准**

    - `N/A`

* **被以下使用**

    - `N/A`

* **支持的平台**

    <table>
    <tr>
        <td>Linux</td>
        <td>macOS</td>
        <td>Windows</td>
    </tr>
    </table>

## <a name="wasmo"></a>[Wasmo](https://github.com/appcypher/wasmo) <sup>[返回顶部⇈](#目录)</sup>
> 基于 LLVM 的 WebAssembly 运行时和编译器.

* **使用语言**

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


* **编译 / 执行模式**

    <table>
    <tr>
        <td>JIT</td>
    </tr>
    </table>

* **与其他语言的互操作性**

    - `N/A`

* **支持的非 MVP 功能**

    - `N/A`

* **支持的主机API**

    - `N/A`

* **非 Web 标准**

    - `N/A`

* **被以下使用**

    - `N/A`

* **支持的平台**

    <table>
    <tr>
        <td>macOS</td>
    </tr>
    </table>

## <a name="wasmrt"></a>[wasmrt](https://github.com/rhitchcock/wasmrt) <sup>[返回顶部⇈](#目录)</sup>
> wasmrt 是一个为本机执行 WebAssembly 模块构建的运行时（首先是虚拟化，最终是 JIT）.

* **使用语言**

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
        <td>解释型</td>
    </tr>
    </table>* **与其他语言的互操作性**

    - `N/A`

* **支持的非 MVP 功能**

    - `N/A`

* **支持的宿主 API**

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

## <a name="wasmtime"></a>[Wasmtime](https://github.com/CraneStation/wasmtime) <sup>[top⇈](#contents)</sup>
> Wasmtime 是一个独立的仅支持 wasm 的 WebAssembly 运行时，使用 Cranelift JIT

* **编写的语言**

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

* **与其他语言的互操作性**

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

* **支持的宿主 API**

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

## <a name="wasmvm1"></a>[WasmVM](https://github.com/WasmVM/WasmVM) <sup>[top⇈](#contents)</sup>
> 一款非官方的独立 WebAssembly 进程虚拟机

* **编写的语言**

    <table>
    <tr>
        <td>C ++</td>
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
        <td>解释型</td>
    </tr>
    </table>

* **与其他语言的互操作性**

    - `N/A`

* **支持的非 MVP 功能**

    - `N/A`* **支持的主机API**

    - `N/A`

* **非网络标准**

    - `N/A`

* **被使用者**

    - `N/A`

* **支持的平台**

    <table>
    <tr>
        <td>Linux</td>
        <td>macOS</td>
        <td>Windows</td>
    </tr>
    </table>

## <a name="wasmvm2"></a>[wasmvm](https://github.com/kogai/wasvm) <sup>[top⇈](#contents)</sup>
> 目标是适用于嵌入式系统的WebAssembly虚拟机。

* **编写的语言**

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


* **编译/执行模式**

    <table>
    <tr>
        <td>解释执行</td>
    </tr>
    </table>

* **与其他语言的互操作性**

    - `N/A`

* **支持的非MVP功能**

    - `N/A`

* **支持的主机API**

    - `N/A`

* **非网络标准**

    - `N/A`

* **被使用者**

    - `N/A`

* **支持的平台**

    <table>
    <tr>
        <td>Linux</td>
        <td>macOS</td>
        <td>Windows</td>
    </tr>
    </table>

## <a name="wavm"></a>[WAVM](https://github.com/WAVM/WAVM) <sup>[top⇈](#contents)</sup>
> 目标是适用于嵌入式系统的WebAssembly虚拟机。

* **编写的语言**

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


* **编译/执行模式**

    <table>
    <tr>
        <td>JIT</td>
    </tr>
    </table>

* **与其他语言的互操作性**

    - `N/A`

* **支持的非MVP功能**

    <table>
    <tr>
        <td>线程</td>
        <td>SIMD</td>
        <td>多返回值和块参数</td>
        <td>异常处理</td>
        <td>名称段</td>
        <td>引用类型</td>
        <td>批量内存操作</td>
        <td>符号扩展指令</td>
    </tr>
    </table>

* **支持的主机API**

    <table>
    <tr>
        <td>WASI</td>。 
    </tr>
    </table><td>Emscripten</td>
        <td>WAVIX</td>
    </tr>
    </table>

* **非Web标准**

    - [x] WASI

* **用途**

    - `N/A`

* **支持的平台**

    <table>
    <tr>
        <td>Linux</td>
        <td>macOS</td>
        <td>Windows</td>
    </tr>
    </table>

## <a name="webassembly"></a>[WebAssembly](https://github.com/dcodeIO/webassembly) <sup>[top⇈](#目录)</sup>
> 一个实验性的、最小的工具集和运行时，建立在node之上，以制作和运行WebAssembly模块

* **编写的语言**

    <table>
    <tr>
        <td>JavaScript</td>
    </tr>
    </table>

* **编译框架**

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

    - `N/A`

* **非MVP功能支持**

    - `N/A`

* **支持的主机API**

    - `N/A`

* **非Web标准**

    - `N/A`

* **用途**

    - `N/A`

* **支持的平台**

    <table>
    <tr>
        <td>Linux</td>
        <td>macOS</td>
        <td>Windows</td>
    </tr>
    </table>


## <a name="wamr"></a>[WAMR](https://github.com/bytecodealliance/wasm-micro-runtime) <sup>[top⇈](#目录)</sup>
> WebAssembly Micro Runtime（WAMR）是一个独立的WebAssembly（WASM）运行时，具有小型占用空间

* **编写的语言**

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
        <td>解释</td>
        <td>AOT-模块</td>
        <td>JIT</td>
    </tr>
    </table>

* **与其他语言的互操作性**

    - `N/A`

* **非MVP功能支持**

    - [非陷阱浮点数向整数转换](https://github.com/WebAssembly/nontrapping-float-to-int-conversions)
    - [扩展操作符](https://github.com/WebAssembly/sign-extension-ops)- [批量内存操作](https://github.com/WebAssembly/bulk-memory-operations)
    - [共享内存](https://github.com/WebAssembly/threads/blob/main/proposals/threads/Overview.md#shared-linear-memory)
    - [多值](https://github.com/WebAssembly/multi-value)
    - [尾调用](https://github.com/WebAssembly/tail-call)


* **支持的主机 API**

    - [wasm-c-api](https://github.com/WebAssembly/wasm-c-api)

* **非 web 标准**

    - [x] WASI

* **被使用于**

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



## <a name="twvm"></a>[TWVM](https://github.com/Becavalier/TWVM) <sup>[返回顶部⇈](#contents)</sup>
>一种小型而高效的 WebAssembly 虚拟机。

* **使用的编程语言**

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
        <td>解释运行</td>
    </tr>
    </table>

* **与其他编程语言的互操作性**

    - `N/A`

* **支持的非 MVP（最小可行产品）功能特性**

    - `N/A`

* **支持的主机 API**

    - `N/A`

* **非 web 标准**

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



## <a name="wazero"></a>[wazero](https://wazero.io) <sup>[返回顶部⇈](#contents)</sup>
> wazero 是一个符合 WebAssembly 1.0 和 2.0 规范的运行时，使用 Go 编写，不依赖平台。

* **使用的编程语言**

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


* **编译 / 执行模式**

    <table>
    <tr><table>
    <tr>
        <td>解释执行</td>
        <td>JIT</td>
    </tr>
</table>

* **与其他语言的互操作性**

    - `不适用`

* **支持的非MVP功能**

    <table>
    <tr>
        <td>大数据集合操作</td>
        <td>导入/导出可变全局对象</td>
        <td>符号扩展操作符</td>
        <td>多返回值</td>
        <td>名称区域</td>
        <td>不会产生异常的浮点数转整数</td>
        <td>引用类型</td>
        <td>SIMD</td>
    </tr>
    </table>

* **支持的 Host APIs**

    <table>
    <tr>
        <td>AssemblyScript</td>
        <td>WASI</td>
    </tr>
    </table>

* **非 Web 标准**

    <table>
    <tr>
        <td>WASI</td>
    </tr>
    </table>

* **被以下项目使用**

    - [dapr](https://github.com/dapr/dapr) - 简化微服务连接的API
    - [trivy](https://github.com/aquasecurity/trivy) - 针对容器和其他构件的漏洞/配置/密钥扫描程序
    - [wapc-go](https://github.com/wapc/wapc-go) - 快速部署 WebAssembly 主机运行时到 waPC-compliant 模块

* **支持的平台**

    <table>
    <tr>
        <td>FreeBSD (amd64)</td>
        <td>Linux (amd64，arm64，riscv64)</td>
        <td>macOs (amd64)</td>
        <td>Windows (amd64)</td>
    </tr>
    </table>
-------------------

## 授权许可

[![CC0](https://licensebuttons.net/l/by/3.0/cn/88x31.png)](https://creativecommons.org/publicdomain/zero/1.0/deed.zh)

在法律允许的范围内，[Steve Akinyemi](https://github.com/appcypher)已放弃对此作品的所有版权及相关或相邻权利。
