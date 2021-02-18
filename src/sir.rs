//! Serialised Intermediate Representation (SIR).
//!
//! SIR is built in-memory during code-generation, and finally placed into an ELF section at link time.

use crate::common::FunctionCx;
use crate::{BackendConfig, CodegenCx, TracerMode};
use cranelift_module::Module;
use object::{write::SymbolSection, SymbolFlags};
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::fx::FxHasher;
use rustc_hir;
use rustc_hir::def_id::LOCAL_CRATE;
use rustc_middle::mir;
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_middle::ty::AdtDef;
use rustc_middle::ty::TyCtxt;
use rustc_middle::ty::TypeFoldable;
use rustc_middle::ty::{self, layout::TyAndLayout};
use rustc_middle::ty::{Instance, Ty};
use rustc_middle::ty::{IntTy, UintTy};
use rustc_span::symbol::Symbol;
use rustc_target::abi::{FieldsShape, LayoutOf, VariantIdx};
use std::alloc::Layout;
use std::convert::TryFrom;
use std::convert::TryInto;
use std::default::Default;
use std::hash::{Hash, Hasher};

const SIR_GLOBAL_SYM_PREFIX: &str = ".yksir";

/// Writes the SIR into a buffer which will be linked in into an ELF section via LLVM.
/// This is based on write_compressed_metadata().
pub(crate) fn write_sir<'tcx>(
    tcx: TyCtxt<'tcx>,
    product: &mut cranelift_object::ObjectProduct,
    cgu_name: &str,
    sir_types: SirTypes,
    sir_funcs: Vec<ykpack::Body>,
) {
    let mut data_buf = Vec::new();
    let mut encoder = ykpack::Encoder::from(&mut data_buf);
    let mut hdr = ykpack::SirHeader::new(sir_types.cgu_hash);

    // First we serialise the types which will be referenced in the body packs that will follow.
    // The serialisation order matters here, as the load order (in the runtime) corresponds with
    // the type indices, hence use of `IndexMap` for insertion order.
    for (typ, typ_idx) in sir_types.map {
        debug_assert!(usize::try_from(typ_idx.0).unwrap() == hdr.types.len());
        hdr.types.push(encoder.tell());
        encoder.serialise(ykpack::Pack::Type(typ)).unwrap();
    }

    for func in sir_funcs {
        hdr.bodies.insert(func.symbol_name.clone(), encoder.tell());
        encoder.serialise(ykpack::Pack::Body(func)).unwrap();
    }

    // Now we encode the header and prepend it to what we encoded above.
    // All offsets are therefore relative to the end of the header.
    let mut buf = Vec::new();
    let mut hdr_encoder = ykpack::Encoder::from(&mut buf);
    hdr_encoder.serialise(ykpack::Pack::Header(hdr)).unwrap();
    buf.append(&mut data_buf);

    // Borrowed from exported_symbols::metadata_symbol_name().
    let symbol_name = format!(
        "{}_{}_{}_{}_sym",
        SIR_GLOBAL_SYM_PREFIX,
        tcx.original_crate_name(LOCAL_CRATE),
        tcx.crate_disambiguator(LOCAL_CRATE)
            .to_fingerprint()
            .to_hex(),
        cgu_name,
    );

    let section_name = format!(
        "{}_{}_{}_{}",
        SIR_GLOBAL_SYM_PREFIX,
        tcx.original_crate_name(LOCAL_CRATE),
        tcx.crate_disambiguator(LOCAL_CRATE)
            .to_fingerprint()
            .to_hex(),
        cgu_name,
    );

    let segment = product
        .object
        .segment_name(object::write::StandardSegment::Data)
        .to_vec();
    let section_id = product.object.add_section(
        segment,
        section_name.into_bytes(),
        object::SectionKind::Data,
    );
    let offset = product.object.append_section_data(section_id, &buf, 1);
    // For MachO and probably PE this is necessary to prevent the linker from throwing away the
    // .rustc section. For ELF this isn't necessary, but it also doesn't harm.
    product.object.add_symbol(object::write::Symbol {
        name: symbol_name.into_bytes(),
        value: offset,
        size: buf.len() as u64,
        kind: object::SymbolKind::Data,
        scope: object::SymbolScope::Dynamic,
        weak: false,
        section: SymbolSection::Section(section_id),
        flags: SymbolFlags::None,
    });
}

impl<'tcx, M: Module> CodegenCx<'tcx, M> {
    fn define_sir_type(&self, ty: ykpack::Ty) -> ykpack::TypeId {
        let mut types = self.sir.as_ref().unwrap().types.borrow_mut();
        ykpack::TypeId {
            cgu: types.cgu_hash,
            idx: types.index(ty),
        }
    }

    pub(crate) fn define_function_sir(&self, sir: ykpack::Body) {
        self.sir.as_ref().unwrap().funcs.borrow_mut().push(sir);
    }

    fn get_size_align(&self, tyid: ykpack::TypeId) -> (usize, usize) {
        let types = self.sir.as_ref().unwrap().types.borrow();
        let ty = types.get(tyid);
        (ty.size, ty.align)
    }
}

pub(crate) const BUILD_SCRIPT_CRATE: &str = "build_script_build";
const CHECKABLE_BINOPS: [ykpack::BinOp; 5] = [
    ykpack::BinOp::Add,
    ykpack::BinOp::Sub,
    ykpack::BinOp::Mul,
    ykpack::BinOp::Shl,
    ykpack::BinOp::Shr,
];

// Generates a big `match` statement for the binary operation lowerings.
macro_rules! binop_lowerings {
    ( $the_op:expr, $($op:ident ),* ) => {
        match $the_op {
            $(mir::BinOp::$op => ykpack::BinOp::$op,)*
        }
    }
}

pub(crate) use ykpack;
pub(crate) use ykpack::build::{Sir, SirTypes};

pub(crate) fn new_sir(tcx: TyCtxt<'_>, cgu_name: &str) -> Sir {
    // Build the CGU hash.
    //
    // This must be a globally unique hash for this compilation unit. It might have been
    // tempting to use the `tcx.crate_hash()` as part of the CGU hash, but this query is
    // invalidated on every source code change to the crate. In turn, that would mean lots of
    // unnecessary rebuilds.
    //
    // We settle on:
    // CGU hash = crate name + crate disambiguator + codegen unit name.
    let mut cgu_hasher = FxHasher::default();
    tcx.crate_name(LOCAL_CRATE).hash(&mut cgu_hasher);
    tcx.crate_disambiguator(LOCAL_CRATE).hash(&mut cgu_hasher);
    cgu_name.hash(&mut cgu_hasher);

    Sir::new(ykpack::CguHash(cgu_hasher.finish()))
}

/// Returns `true` if we should collect SIR for the current crate.
pub(crate) fn is_sir_required(tcx: TyCtxt<'_>, config: BackendConfig) -> bool {
    !matches!(config.tracer_mode, TracerMode::None)
        && tcx.crate_name(LOCAL_CRATE).as_str() != BUILD_SCRIPT_CRATE
}

/// A structure for building the SIR of a function.
pub(crate) struct SirFuncCx<'tcx> {
    /// The instance we are lowering.
    instance: Instance<'tcx>,
    /// The MIR body of the above instance.
    mir: &'tcx mir::Body<'tcx>,
    /// The builder for the SIR function.
    pub(crate) sir_builder: ykpack::build::SirBuilder,
    /// Maps each MIR local to a SIR IPlace.
    var_map: FxHashMap<mir::Local, ykpack::IPlace>,
    /// The compiler's type context.
    tcx: TyCtxt<'tcx>,
}

impl<'tcx> SirFuncCx<'tcx> {
    pub(crate) fn new(fx: &FunctionCx<'_, 'tcx, impl Module>) -> Self {
        let symbol_name = String::from(&*fx.tcx.symbol_name(fx.instance).name);

        let mut flags = ykpack::BodyFlags::empty();
        for attr in fx.tcx.get_attrs(fx.instance.def_id()).iter() {
            if fx.tcx.sess.check_name(attr, Symbol::intern("do_not_trace")) {
                flags |= ykpack::BodyFlags::DO_NOT_TRACE;
            } else if fx.tcx.sess.check_name(attr, Symbol::intern("interp_step")) {
                // Check various properties of the interp_step at compile time.
                if !fx.tcx.upvars_mentioned(fx.instance.def_id()).is_none() {
                    fx.tcx.sess.span_fatal(
                        fx.tcx.def_span(fx.instance.def_id()),
                        "The #[interp_step] function must not capture from its environment",
                    );
                }

                if fx.mir.args_iter().count() != 1 {
                    fx.tcx.sess.span_fatal(
                        fx.tcx.def_span(fx.instance.def_id()),
                        "The #[interp_step] function must accept only one argument",
                    );
                }

                let arg_ok = if let ty::Ref(_, inner_ty, rustc_hir::Mutability::Mut) =
                    fx.mir.local_decls[mir::Local::from_u32(1)].ty.kind()
                {
                    if let ty::Adt(def, _) = inner_ty.kind() {
                        def.is_struct()
                    } else {
                        false
                    }
                } else {
                    false
                };
                if !arg_ok {
                    fx.tcx.sess.span_err(
                        fx.tcx.def_span(fx.instance.def_id()),
                        "The #[interp_step] function must accept a mutable reference to a struct",
                    );
                }

                if !fx.mir.return_ty().is_unit() {
                    fx.tcx.sess.span_err(
                        fx.tcx.def_span(fx.instance.def_id()),
                        "The #[interp_step] function must return unit",
                    );
                }

                flags |= ykpack::BodyFlags::INTERP_STEP;
            } else if fx.tcx.sess.check_name(attr, Symbol::intern("trace_debug")) {
                flags |= ykpack::BodyFlags::TRACE_DEBUG;
            }
        }

        let crate_name = fx.tcx.crate_name(fx.instance.def_id().krate).as_str();
        if crate_name == "core" || crate_name == "alloc" {
            flags |= ykpack::BodyFlags::DO_NOT_TRACE;
        }
        let var_map: FxHashMap<mir::Local, ykpack::IPlace> = FxHashMap::default();

        let mut this = Self {
            instance: fx.instance.clone(),
            mir: fx.mir,
            sir_builder: ykpack::build::SirBuilder::new(
                symbol_name,
                flags,
                fx.mir.arg_count,
                fx.mir.basic_blocks().len(),
            ),
            var_map,
            tcx: fx.tcx,
        };

        // Allocate return local and args in their anchored positions.
        for idx in 0..=fx.mir.arg_count {
            let ml = mir::Local::from_usize(idx);
            let sirty =
                this.lower_ty_and_layout(fx, &this.mono_layout_of(fx, this.mir.local_decls[ml].ty));
            this.sir_builder.func.local_decls.push(ykpack::LocalDecl {
                ty: sirty,
                referenced: false,
            });
            this.var_map.insert(
                ml,
                ykpack::IPlace::Val {
                    local: ykpack::Local(u32::try_from(idx).unwrap()),
                    off: 0,
                    ty: sirty,
                },
            );
        }
        this
    }

    /// Compute layout and offsets required for blackholing.
    pub(crate) fn compute_layout_and_offsets(&mut self, fx: &FunctionCx<'_, 'tcx, impl Module>) {
        let mut layout = Layout::from_size_align(0, 1).unwrap();
        for ld in &self.sir_builder.func.local_decls {
            let (size, align) = fx.cx.get_size_align(ld.ty);
            let l = Layout::from_size_align(size, align).unwrap();
            let (nl, off) = layout.extend(l).unwrap();
            self.sir_builder.func.offsets.push(off);
            layout = nl;
        }
        layout = layout.pad_to_align();
        self.sir_builder.func.layout = (layout.size(), layout.align());
    }

    /// Returns the IPlace corresponding with MIR local `ml`. A new IPlace is constructed if we've
    /// never seen this MIR local before.
    fn sir_local(
        &mut self,
        fx: &FunctionCx<'_, 'tcx, impl Module>,
        ml: &mir::Local,
    ) -> ykpack::IPlace {
        let ret = if let Some(ip) = self.var_map.get(ml) {
            ip.clone()
        } else {
            let sirty = self
                .lower_ty_and_layout(fx, &self.mono_layout_of(fx, self.mir.local_decls[*ml].ty));
            let nl = self.new_sir_local(sirty);
            self.var_map.insert(*ml, nl.clone());
            nl
        };
        ret
    }

    /// Returns a zero-offset IPlace for a new SIR local.
    fn new_sir_local(&mut self, sirty: ykpack::TypeId) -> ykpack::IPlace {
        self.sir_builder.new_sir_local(sirty)
    }

    /// Tells the tracer codegen that the local `l` is referenced, and that is should be allocated
    /// directly to the stack and not a register. You can't reference registers.
    fn notify_referenced(&mut self, l: ykpack::Local) {
        self.sir_builder.notify_referenced(l)
    }

    /// Returns true if there are no basic blocks.
    pub(crate) fn is_empty(&self) -> bool {
        self.sir_builder.is_empty()
    }

    /// Appends a statement to the specified basic block.
    fn push_stmt(&mut self, bb: ykpack::BasicBlockIndex, stmt: ykpack::Statement) {
        self.sir_builder.push_stmt(bb, stmt)
    }

    /// Sets the terminator of the specified block.
    pub(crate) fn set_terminator(
        &mut self,
        bb: ykpack::BasicBlockIndex,
        new_term: ykpack::Terminator,
    ) {
        self.sir_builder.set_terminator(bb, new_term);
    }

    pub(crate) fn set_term_switchint(
        &mut self,
        fx: &FunctionCx<'_, 'tcx, impl Module>,
        bb: ykpack::BasicBlockIndex,
        discr: &mir::Operand<'tcx>,
        targets: &mir::SwitchTargets,
    ) {
        let mut values = Vec::new();
        let mut target_bbs = Vec::new();
        for (v, t) in targets.iter() {
            values.push(v);
            target_bbs.push(t.as_u32());
        }
        let new_term = ykpack::Terminator::SwitchInt {
            discr: self.lower_operand(fx, bb, discr),
            values,
            target_bbs,
            otherwise_bb: targets.otherwise().as_u32(),
        };
        self.set_terminator(bb, new_term);
    }

    pub(crate) fn set_term_goto(&mut self, bb: mir::BasicBlock, target: mir::BasicBlock) {
        self.set_terminator(bb.as_u32(), ykpack::Terminator::Goto(target.as_u32()));
    }

    pub(crate) fn set_term_assert(
        &mut self,
        fx: &FunctionCx<'_, 'tcx, impl Module>,
        bb: mir::BasicBlock,
        cond: &mir::Operand<'tcx>,
        expected: bool,
        target_bb: mir::BasicBlock,
    ) {
        let bb = bb.as_u32();
        let cond_ip = self.lower_operand(fx, bb, cond);
        let term = ykpack::Terminator::Assert {
            cond: cond_ip,
            expected,
            target_bb: target_bb.as_u32(),
        };
        self.set_terminator(bb, term);
    }

    /// Converts a MIR statement to SIR, appending the result to `bb`.
    pub(crate) fn lower_statement(
        &mut self,
        fx: &FunctionCx<'_, 'tcx, impl Module>,
        bb: ykpack::BasicBlockIndex,
        stmt: &mir::Statement<'tcx>,
    ) {
        match stmt.kind {
            mir::StatementKind::Assign(box (ref place, ref rvalue)) => {
                self.lower_assign_stmt(fx, bb, place, rvalue)
            }
            mir::StatementKind::StorageLive(local) => {
                let local = self.sir_local(fx, &local).local().unwrap();
                self.push_stmt(bb, ykpack::Statement::StorageLive(local))
            }
            mir::StatementKind::StorageDead(local) => {
                let local = self.sir_local(fx, &local).local().unwrap();
                self.push_stmt(bb, ykpack::Statement::StorageDead(local))
            }
            _ => self.push_stmt(bb, ykpack::Statement::Unimplemented(format!("{:?}", stmt))),
        }
    }

    fn lower_assign_stmt(
        &mut self,
        fx: &FunctionCx<'_, 'tcx, impl Module>,
        bb: ykpack::BasicBlockIndex,
        lvalue: &mir::Place<'tcx>,
        rvalue: &mir::Rvalue<'tcx>,
    ) {
        let dest_ty = lvalue.ty(self.mir, self.tcx).ty;
        let rhs = self.lower_rvalue(fx, bb, dest_ty, rvalue);

        // FIXME optimisation.
        // If the store can't affect any state observable from outside the function, then don't
        // emit a store, but instead just update the mapping. This will remove many unnecessary
        // stores and also act as a kind of constant propagation.
        let lhs = self.lower_place(fx, bb, lvalue);
        self.push_stmt(bb, ykpack::Statement::Store(lhs, rhs));
    }

    pub(crate) fn lower_operand(
        &mut self,
        fx: &FunctionCx<'_, 'tcx, impl Module>,
        bb: ykpack::BasicBlockIndex,
        operand: &mir::Operand<'tcx>,
    ) -> ykpack::IPlace {
        match operand {
            mir::Operand::Copy(place) | mir::Operand::Move(place) => {
                self.lower_place(fx, bb, place)
            }
            mir::Operand::Constant(cst) => self.lower_constant(fx, cst),
        }
    }

    fn lower_cast_misc(
        &mut self,
        fx: &FunctionCx<'_, 'tcx, impl Module>,
        bb: ykpack::BasicBlockIndex,
        op: &mir::Operand<'tcx>,
        ty: Ty<'tcx>,
    ) -> ykpack::IPlace {
        let lop = self.lower_operand(fx, bb, op);

        // The ty we are casting to is equivalent to dest_ty.
        let ty = self.lower_ty_and_layout(fx, &self.mono_layout_of(fx, ty));
        let dest_ip = self.new_sir_local(ty);
        let stmt = ykpack::Statement::Cast(dest_ip.clone(), lop);
        self.push_stmt(bb, stmt);
        dest_ip
    }

    fn lower_rvalue(
        &mut self,
        fx: &FunctionCx<'_, 'tcx, impl Module>,
        bb: ykpack::BasicBlockIndex,
        dest_ty: Ty<'tcx>,
        rvalue: &mir::Rvalue<'tcx>,
    ) -> ykpack::IPlace {
        match rvalue {
            mir::Rvalue::Use(opnd) => self.lower_operand(fx, bb, opnd),
            mir::Rvalue::Ref(_, _, p) => self.lower_ref(fx, bb, dest_ty, p),
            mir::Rvalue::BinaryOp(op, opnd1, opnd2) => {
                self.lower_binop(fx, bb, dest_ty, *op, opnd1, opnd2, false)
            }
            mir::Rvalue::CheckedBinaryOp(op, opnd1, opnd2) => {
                self.lower_binop(fx, bb, dest_ty, *op, opnd1, opnd2, true)
            }
            mir::Rvalue::Cast(mir::CastKind::Misc, op, ty) => self.lower_cast_misc(fx, bb, op, ty),
            mir::Rvalue::Len(p) => {
                let ip = self.lower_place(fx, bb, p);
                match p.ty(&self.mir.local_decls, self.tcx).ty.kind() {
                    ty::Array(_elem_ty, len) => {
                        let raw_val =
                            usize::try_from(len.eval_usize(self.tcx, ty::ParamEnv::reveal_all()))
                                .unwrap();
                        let val = ykpack::Constant::Int(ykpack::ConstantInt::UnsignedInt(
                            ykpack::UnsignedInt::Usize(raw_val),
                        ));
                        ykpack::IPlace::Const { val, ty: ip.ty() }
                    }
                    ty::Slice(_elem_ty) => self.offset_iplace(
                        fx,
                        ip,
                        i32::try_from(self.tcx.data_layout.pointer_size.bytes()).unwrap(),
                        dest_ty,
                    ),
                    _ => unreachable!(),
                }
            }
            _ => ykpack::IPlace::Unimplemented(with_no_trimmed_paths(|| {
                format!("unimplemented rvalue: {:?}", rvalue)
            })),
        }
    }

    fn monomorphize<T>(&self, value: &T) -> T
    where
        T: TypeFoldable<'tcx> + Copy,
    {
        self.instance.subst_mir_and_normalize_erasing_regions(
            self.tcx,
            ty::ParamEnv::reveal_all(),
            *value,
        )
    }

    /// Wrapper for bx.layout_of() which ensures the type is first monomorphised.
    fn mono_layout_of(
        &self,
        fx: &FunctionCx<'_, 'tcx, impl Module>,
        t: Ty<'tcx>,
    ) -> TyAndLayout<'tcx> {
        fx.layout_of(fx.monomorphize(&t))
    }

    /// Apply an offset to an IPlace.
    fn offset_iplace(
        &mut self,
        fx: &FunctionCx<'_, 'tcx, impl Module>,
        mut ip: ykpack::IPlace,
        add: ykpack::OffT,
        mirty: Ty<'tcx>,
    ) -> ykpack::IPlace {
        match &mut ip {
            ykpack::IPlace::Val { off, ty, .. } => {
                *off += add;
                *ty = self.lower_ty_and_layout(fx, &self.mono_layout_of(fx, mirty));
                ip
            }
            ykpack::IPlace::Indirect { off, ty, .. } => {
                *off += add;
                *ty = self.lower_ty_and_layout(fx, &self.mono_layout_of(fx, mirty));
                ip
            }
            ykpack::IPlace::Const { .. } => {
                ykpack::IPlace::Unimplemented("offset_iplace on a constant".to_owned())
            }
            ykpack::IPlace::Unimplemented(_) => ip,
        }
    }

    pub(crate) fn lower_place(
        &mut self,
        fx: &FunctionCx<'_, 'tcx, impl Module>,
        bb: ykpack::BasicBlockIndex,
        place: &mir::Place<'_>,
    ) -> ykpack::IPlace {
        // We start with the base local and project away from it.
        let mut cur_iplace = self.sir_local(fx, &place.local);
        let mut cur_mirty = self.monomorphize(&self.mir.local_decls[place.local].ty);

        // Loop over the projection chain, updating cur_iplace as we go.
        for pj in place.projection {
            let next_mirty = match pj {
                mir::ProjectionElem::Field(f, _) => {
                    let fi = f.as_usize();
                    match cur_mirty.kind() {
                        ty::Adt(def, _) => {
                            if def.is_struct() {
                                let ty_lay = self.mono_layout_of(fx, cur_mirty);
                                let st_lay = ty_lay.for_variant(fx, VariantIdx::from_u32(0));
                                if let FieldsShape::Arbitrary { offsets, .. } = &st_lay.fields {
                                    let new_mirty = st_lay.field(fx, fi).ty;
                                    cur_iplace = self.offset_iplace(
                                        fx,
                                        cur_iplace,
                                        offsets[fi].bytes().try_into().unwrap(),
                                        new_mirty,
                                    );
                                    new_mirty
                                } else {
                                    return ykpack::IPlace::Unimplemented(format!(
                                        "struct field shape: {:?}",
                                        st_lay.fields
                                    ));
                                }
                            } else if def.is_enum() {
                                return ykpack::IPlace::Unimplemented(format!(
                                    "enum_projection: {:?}",
                                    def
                                ));
                            } else {
                                return ykpack::IPlace::Unimplemented(format!("adt: {:?}", def));
                            }
                        }
                        ty::Tuple(..) => {
                            let tup_lay = self.mono_layout_of(fx, cur_mirty);
                            match &tup_lay.fields {
                                FieldsShape::Arbitrary { offsets, .. } => {
                                    let new_mirty = tup_lay.field(fx, fi).ty;
                                    cur_iplace = self.offset_iplace(
                                        fx,
                                        cur_iplace,
                                        offsets[fi].bytes().try_into().unwrap(),
                                        new_mirty,
                                    );
                                    new_mirty
                                }
                                _ => {
                                    return ykpack::IPlace::Unimplemented(format!(
                                        "tuple field shape: {:?}",
                                        tup_lay.fields
                                    ));
                                }
                            }
                        }
                        _ => {
                            return ykpack::IPlace::Unimplemented(format!(
                                "field access on: {:?}",
                                cur_mirty
                            ));
                        }
                    }
                }
                mir::ProjectionElem::Index(idx) => match cur_mirty.kind() {
                    ty::Array(elem_ty, ..) | ty::Slice(elem_ty) => {
                        let arr_lay = self.mono_layout_of(fx, cur_mirty);
                        let elem_size = match &arr_lay.fields {
                            FieldsShape::Array { stride, .. } => {
                                u32::try_from(stride.bytes_usize()).unwrap()
                            }
                            _ => unreachable!(),
                        };

                        let dest_ty =
                            self.lower_ty_and_layout(fx, &self.mono_layout_of(fx, elem_ty));
                        let dest = self.new_sir_local(dest_ty);
                        let idx_ip = self.sir_local(fx, &idx);
                        let stmt = ykpack::Statement::DynOffs {
                            dest: dest.clone(),
                            base: cur_iplace.clone(),
                            idx: idx_ip,
                            scale: elem_size,
                        };
                        self.push_stmt(bb, stmt);
                        cur_iplace = dest.to_indirect(dest_ty);
                        elem_ty
                    }
                    _ => {
                        return ykpack::IPlace::Unimplemented(format!("index on {:?}", cur_mirty));
                    }
                },
                mir::ProjectionElem::Deref => {
                    match cur_mirty.kind() {
                        ty::Ref(_, ty, _) | ty::RawPtr(ty::TypeAndMut { ty, .. }) => {
                            if let ykpack::IPlace::Indirect { ty: dty, .. } = cur_iplace {
                                // We are dereffing an already indirect place, so we emit an
                                // intermediate store to strip away one level of indirection.
                                let dest = self.new_sir_local(dty);
                                let deref =
                                    ykpack::Statement::Store(dest.clone(), cur_iplace.clone());
                                self.push_stmt(bb, deref);
                                cur_iplace = dest;
                            }

                            if let Some(l) = cur_iplace.local() {
                                self.notify_referenced(l);
                            }

                            let tyid = self.lower_ty_and_layout(fx, &self.mono_layout_of(fx, ty));
                            cur_iplace = cur_iplace.to_indirect(tyid);
                            ty
                        }
                        _ => {
                            return ykpack::IPlace::Unimplemented(format!(
                                "invalid deref: {:?}",
                                cur_mirty.kind()
                            ));
                        }
                    }
                }
                _ => return ykpack::IPlace::Unimplemented(format!("projection: {:?}", pj)),
            };
            cur_mirty = self.monomorphize(&next_mirty);
        }
        cur_iplace
    }

    fn lower_constant(
        &mut self,
        fx: &FunctionCx<'_, 'tcx, impl Module>,
        constant: &mir::Constant<'tcx>,
    ) -> ykpack::IPlace {
        match constant.literal.val {
            ty::ConstKind::Value(mir::interpret::ConstValue::Scalar(s)) => {
                let val = self.lower_scalar(fx, constant.literal.ty, s);
                let ty =
                    self.lower_ty_and_layout(fx, &self.mono_layout_of(fx, constant.literal.ty));
                ykpack::IPlace::Const { val, ty }
            }
            _ => ykpack::IPlace::Unimplemented(with_no_trimmed_paths(|| {
                format!("unimplemented constant: {:?}", constant)
            })),
        }
    }

    fn lower_scalar(
        &mut self,
        fx: &FunctionCx<'_, 'tcx, impl Module>,
        ty: Ty<'tcx>,
        s: mir::interpret::Scalar,
    ) -> ykpack::Constant {
        match ty.kind() {
            ty::Uint(uint) => self
                .lower_uint(*uint, s)
                .map(|i| ykpack::Constant::Int(ykpack::ConstantInt::UnsignedInt(i)))
                .unwrap_or_else(|_| {
                    with_no_trimmed_paths(|| {
                        ykpack::Constant::Unimplemented(format!(
                            "unimplemented uint scalar: {:?}",
                            ty.kind()
                        ))
                    })
                }),
            ty::Int(int) => self
                .lower_int(*int, s)
                .map(|i| ykpack::Constant::Int(ykpack::ConstantInt::SignedInt(i)))
                .unwrap_or_else(|_| {
                    ykpack::Constant::Unimplemented(format!(
                        "unimplemented signed int scalar: {:?}",
                        ty.kind()
                    ))
                }),
            ty::Bool => self.lower_bool(s),
            ty::Tuple(_) => {
                // FIXME for now just the unit tuple. Need to implement arbitrary scalar tuples.
                if ty.is_unit() {
                    let tyid = self.lower_ty_and_layout(fx, &self.mono_layout_of(fx, ty));
                    ykpack::Constant::Tuple(tyid)
                } else {
                    ykpack::Constant::Unimplemented(format!(
                        "unimplemented scalar: {:?}",
                        ty.kind()
                    ))
                }
            }
            _ => ykpack::Constant::Unimplemented(format!("unimplemented scalar: {:?}", ty.kind())),
        }
    }

    /// Lower an unsigned integer.
    fn lower_uint(
        &self,
        uint: UintTy,
        s: mir::interpret::Scalar,
    ) -> Result<ykpack::UnsignedInt, ()> {
        match uint {
            UintTy::U8 => match s.to_u8() {
                Ok(val) => Ok(ykpack::UnsignedInt::U8(val)),
                Err(e) => panic!("Could not lower scalar to u8: {}", e),
            },
            UintTy::U16 => match s.to_u16() {
                Ok(val) => Ok(ykpack::UnsignedInt::U16(val)),
                Err(e) => panic!("Could not lower scalar to u16: {}", e),
            },
            UintTy::U32 => match s.to_u32() {
                Ok(val) => Ok(ykpack::UnsignedInt::U32(val)),
                Err(e) => panic!("Could not lower scalar to u32: {}", e),
            },
            UintTy::U64 => match s.to_u64() {
                Ok(val) => Ok(ykpack::UnsignedInt::U64(val)),
                Err(e) => panic!("Could not lower scalar to u64: {}", e),
            },
            UintTy::Usize => match s.to_machine_usize(&self.tcx) {
                Ok(val) => Ok(ykpack::UnsignedInt::Usize(val as usize)),
                Err(e) => panic!("Could not lower scalar to usize: {}", e),
            },
            _ => Err(()),
        }
    }

    /// Lower a signed integer.
    fn lower_int(&self, int: IntTy, s: mir::interpret::Scalar) -> Result<ykpack::SignedInt, ()> {
        match int {
            IntTy::I8 => match s.to_i8() {
                Ok(val) => Ok(ykpack::SignedInt::I8(val)),
                Err(e) => panic!("Could not lower scalar to i8: {}", e),
            },
            IntTy::I16 => match s.to_i16() {
                Ok(val) => Ok(ykpack::SignedInt::I16(val)),
                Err(e) => panic!("Could not lower scalar to i16: {}", e),
            },
            IntTy::I32 => match s.to_i32() {
                Ok(val) => Ok(ykpack::SignedInt::I32(val)),
                Err(e) => panic!("Could not lower scalar to i32: {}", e),
            },
            IntTy::I64 => match s.to_i64() {
                Ok(val) => Ok(ykpack::SignedInt::I64(val)),
                Err(e) => panic!("Could not lower scalar to i64: {}", e),
            },
            IntTy::Isize => match s.to_machine_isize(&self.tcx) {
                Ok(val) => Ok(ykpack::SignedInt::Isize(val as isize)),
                Err(e) => panic!("Could not lower scalar to isize: {}", e),
            },
            _ => Err(()),
        }
    }

    fn lower_binop(
        &mut self,
        fx: &FunctionCx<'_, 'tcx, impl Module>,
        bb: ykpack::BasicBlockIndex,
        dest_ty: Ty<'tcx>,
        op: mir::BinOp,
        opnd1: &mir::Operand<'tcx>,
        opnd2: &mir::Operand<'tcx>,
        checked: bool,
    ) -> ykpack::IPlace {
        let op = binop_lowerings!(
            op, Add, Sub, Mul, Div, Rem, BitXor, BitAnd, BitOr, Shl, Shr, Eq, Lt, Le, Ne, Ge, Gt,
            Offset
        );
        let opnd1 = self.lower_operand(fx, bb, opnd1);
        let opnd2 = self.lower_operand(fx, bb, opnd2);

        if checked {
            debug_assert!(CHECKABLE_BINOPS.contains(&op));
        }

        let ty = self.lower_ty_and_layout(fx, &self.mono_layout_of(fx, dest_ty));
        let dest_ip = self.new_sir_local(ty);
        let stmt = ykpack::Statement::BinaryOp {
            dest: dest_ip.clone(),
            op,
            opnd1,
            opnd2,
            checked,
        };
        self.push_stmt(bb, stmt);
        dest_ip
    }

    fn lower_bool(&self, s: mir::interpret::Scalar) -> ykpack::Constant {
        match s.to_bool() {
            Ok(val) => ykpack::Constant::Bool(val),
            Err(e) => panic!("Could not lower scalar (bool) to u8: {}", e),
        }
    }

    fn lower_ref(
        &mut self,
        fx: &FunctionCx<'_, 'tcx, impl Module>,
        bb: ykpack::BasicBlockIndex,
        dest_ty: Ty<'tcx>,
        place: &mir::Place<'tcx>,
    ) -> ykpack::IPlace {
        let ty = self.lower_ty_and_layout(fx, &self.mono_layout_of(fx, dest_ty));
        let dest_ip = self.new_sir_local(ty);
        let src_ip = self.lower_place(fx, bb, place);
        let mkref = ykpack::Statement::MkRef(dest_ip.clone(), src_ip.clone());
        if let Some(src_local) = src_ip.local() {
            self.notify_referenced(src_local);
        }
        self.push_stmt(bb, mkref);
        dest_ip
    }

    fn lower_ty_and_layout(
        &mut self,
        fx: &FunctionCx<'_, 'tcx, impl Module>,
        ty_layout: &TyAndLayout<'tcx>,
    ) -> ykpack::TypeId {
        let size = usize::try_from(ty_layout.layout.size.bytes()).unwrap();
        let align = usize::try_from(ty_layout.layout.align.abi.bytes()).unwrap();
        let sir_tykind = match ty_layout.ty.kind() {
            ty::Int(si) => self.lower_signed_int_ty(*si),
            ty::Uint(ui) => self.lower_unsigned_int_ty(*ui),
            ty::Adt(adt_def, ..) => self.lower_adt_ty(fx, adt_def, &ty_layout),
            ty::Array(elem_ty, len) => ykpack::TyKind::Array {
                elem_ty: self.lower_ty_and_layout(fx, &self.mono_layout_of(fx, elem_ty)),
                len: usize::try_from(len.eval_usize(self.tcx, ty::ParamEnv::reveal_all())).unwrap(),
            },
            ty::Slice(typ) => {
                ykpack::TyKind::Slice(self.lower_ty_and_layout(fx, &self.mono_layout_of(fx, typ)))
            }
            ty::Ref(_, typ, _) => {
                ykpack::TyKind::Ref(self.lower_ty_and_layout(fx, &self.mono_layout_of(fx, typ)))
            }
            ty::Bool => ykpack::TyKind::Bool,
            ty::Char => ykpack::TyKind::Char,
            ty::Tuple(..) => self.lower_tuple_ty(fx, ty_layout),
            _ => ykpack::TyKind::Unimplemented(format!("{:?}", ty_layout)),
        };
        let sir_ty = ykpack::Ty {
            size,
            align,
            kind: sir_tykind,
        };
        let tyid = fx.cx.define_sir_type(sir_ty);
        tyid
    }

    fn lower_signed_int_ty(&mut self, si: IntTy) -> ykpack::TyKind {
        match si {
            IntTy::Isize => ykpack::TyKind::SignedInt(ykpack::SignedIntTy::Isize),
            IntTy::I8 => ykpack::TyKind::SignedInt(ykpack::SignedIntTy::I8),
            IntTy::I16 => ykpack::TyKind::SignedInt(ykpack::SignedIntTy::I16),
            IntTy::I32 => ykpack::TyKind::SignedInt(ykpack::SignedIntTy::I32),
            IntTy::I64 => ykpack::TyKind::SignedInt(ykpack::SignedIntTy::I64),
            IntTy::I128 => ykpack::TyKind::SignedInt(ykpack::SignedIntTy::I128),
        }
    }

    fn lower_unsigned_int_ty(&mut self, ui: UintTy) -> ykpack::TyKind {
        match ui {
            UintTy::Usize => ykpack::TyKind::UnsignedInt(ykpack::UnsignedIntTy::Usize),
            UintTy::U8 => ykpack::TyKind::UnsignedInt(ykpack::UnsignedIntTy::U8),
            UintTy::U16 => ykpack::TyKind::UnsignedInt(ykpack::UnsignedIntTy::U16),
            UintTy::U32 => ykpack::TyKind::UnsignedInt(ykpack::UnsignedIntTy::U32),
            UintTy::U64 => ykpack::TyKind::UnsignedInt(ykpack::UnsignedIntTy::U64),
            UintTy::U128 => ykpack::TyKind::UnsignedInt(ykpack::UnsignedIntTy::U128),
        }
    }

    fn lower_tuple_ty(
        &mut self,
        fx: &FunctionCx<'_, 'tcx, impl Module>,
        ty_layout: &TyAndLayout<'tcx>,
    ) -> ykpack::TyKind {
        match &ty_layout.fields {
            FieldsShape::Arbitrary { offsets, .. } => {
                let mut sir_offsets = Vec::new();
                let mut sir_tys = Vec::new();
                for (idx, off) in offsets.iter().enumerate() {
                    sir_tys.push(self.lower_ty_and_layout(fx, &ty_layout.field(fx, idx)));
                    sir_offsets.push(off.bytes().try_into().unwrap());
                }

                ykpack::TyKind::Tuple(ykpack::TupleTy {
                    fields: ykpack::Fields {
                        offsets: sir_offsets,
                        tys: sir_tys,
                    },
                })
            }
            _ => ykpack::TyKind::Unimplemented(format!("{:?}", ty_layout)),
        }
    }

    fn lower_adt_ty(
        &mut self,
        fx: &FunctionCx<'_, 'tcx, impl Module>,
        adt_def: &AdtDef,
        ty_layout: &TyAndLayout<'tcx>,
    ) -> ykpack::TyKind {
        if adt_def.variants.len() == 1 {
            // Plain old struct-like thing.
            let struct_layout = ty_layout.for_variant(fx, VariantIdx::from_u32(0));

            match &ty_layout.fields {
                FieldsShape::Arbitrary { offsets, .. } => {
                    let mut sir_offsets = Vec::new();
                    let mut sir_tys = Vec::new();
                    for (idx, off) in offsets.iter().enumerate() {
                        sir_tys.push(self.lower_ty_and_layout(fx, &struct_layout.field(fx, idx)));
                        sir_offsets.push(off.bytes().try_into().unwrap());
                    }

                    ykpack::TyKind::Struct(ykpack::StructTy {
                        fields: ykpack::Fields {
                            offsets: sir_offsets,
                            tys: sir_tys,
                        },
                    })
                }
                _ => ykpack::TyKind::Unimplemented(format!("{:?}", ty_layout)),
            }
        } else {
            // An enum with variants.
            ykpack::TyKind::Unimplemented(format!("{:?}", ty_layout))
        }
    }
}
